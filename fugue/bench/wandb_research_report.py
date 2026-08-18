from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import os
import re
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, cast
from urllib.parse import urlsplit

from fugue.bench.candidates import stable_digest
from fugue.bench.research_report import (
    ResearchIndexReportProjectionV1,
    ResearchIndexReportPublicationOutcomeV1,
    ResearchIndexReportPublisher,
)
from fugue.bench.wandb_research_index import (
    WandbResearchIndexPublicationError,
    wandb_research_index_target_from_environment,
)
from fugue.weave_support import EVIDENCE_ROUTING_LOCK

SUPPORTED_WANDB_VERSION = "0.28.1"
SUPPORTED_WANDB_WORKSPACES_VERSION = "0.4.5"
WANDB_REPORT_API = "wandb-workspaces.reports.v2"
WANDB_REPORT_API_STABILITY = "public_preview"
WANDB_RESEARCH_REPORT_PUBLISHER_ID = "fugue-wandb-research-report"
WANDB_RESEARCH_REPORT_PUBLISHER_REVISION = (
    f"v1+wandb-{SUPPORTED_WANDB_VERSION}"
    f"+wandb-workspaces-{SUPPORTED_WANDB_WORKSPACES_VERSION}"
)

_PROJECT_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MARKER_PREFIX = "fugue.research_report.projection_digest="
_MAX_REPORT_SCAN = 100
_AUTH_ENV = (
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
    "WANDB_APP_BASE_URL",
    "WANDB_APP_URL",
)
_ROUTING_ENV = (
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_RUN_ID",
    "WANDB_NAME",
    "WANDB_JOB_TYPE",
    "WANDB_RUN_GROUP",
)
_SCOPED_ENV = (*_AUTH_ENV, *_ROUTING_ENV)
_PUBLICATION_LOCK = EVIDENCE_ROUTING_LOCK


class MissingWandbReportExtraError(RuntimeError):
    """The pinned optional W&B Report dependencies are unavailable."""


class WandbResearchReportPublicationError(RuntimeError):
    """W&B did not authoritatively preserve the exact Report projection."""


class RetryableWandbResearchReportPublicationError(WandbResearchReportPublicationError):
    """Remote Report finalization or readback is not yet authoritative."""


def wandb_research_report_publisher_from_environment(
    env: Mapping[str, str],
) -> ResearchIndexReportPublisher:
    """Build the optional W&B Report publisher bound to a projection digest.

    Importing this module never imports W&B. The factory loads and checks both
    optional packages before it creates an API client or performs any network
    operation.
    """

    try:
        wandb = importlib.import_module("wandb")
        reports = importlib.import_module("wandb_workspaces.reports.v2")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MissingWandbReportExtraError(
            "W&B Research Report publication requires the pinned optional "
            "W&B SDK and Reports API. Install them with "
            '`python -m pip install "fugue[wandb-report]"`.'
        ) from exc
    _verify_optional_api(wandb=wandb, reports=reports)
    publication_env = {name: str(env[name]) for name in _AUTH_ENV if name in env}

    def publish(
        projection: ResearchIndexReportProjectionV1,
        *,
        expected_report_id: str | None = None,
        expected_report_url: str | None = None,
    ) -> ResearchIndexReportPublicationOutcomeV1:
        if not isinstance(projection, ResearchIndexReportProjectionV1):
            raise WandbResearchReportPublicationError(
                "W&B Research Report publisher requires ResearchIndexReportProjectionV1"
            )
        try:
            expected_target = wandb_research_index_target_from_environment(
                projection.target.project,
                env,
            )
        except WandbResearchIndexPublicationError as exc:
            raise WandbResearchReportPublicationError(
                "W&B Research Report publisher environment is invalid"
            ) from exc
        if projection.target != expected_target:
            raise WandbResearchReportPublicationError(
                "W&B Research Report target does not match the publisher environment"
            )
        return _publish_with_wandb_report_api(
            wandb=wandb,
            reports=reports,
            projection=projection,
            publication_env=publication_env,
            expected_report_id=expected_report_id,
            expected_report_url=expected_report_url,
        )

    return cast(ResearchIndexReportPublisher, publish)


def _verify_optional_api(*, wandb: Any, reports: Any) -> None:
    observed_wandb = str(getattr(wandb, "__version__", "") or "")
    if observed_wandb != SUPPORTED_WANDB_VERSION:
        raise MissingWandbReportExtraError(
            "W&B Research Report publication requires "
            f"wandb=={SUPPORTED_WANDB_VERSION}; found "
            f"{observed_wandb or 'an unknown version'}"
        )
    try:
        observed_workspaces = importlib.metadata.version("wandb-workspaces")
    except importlib.metadata.PackageNotFoundError as exc:
        raise MissingWandbReportExtraError(
            "W&B Research Report publication requires "
            f"wandb-workspaces=={SUPPORTED_WANDB_WORKSPACES_VERSION}"
        ) from exc
    if observed_workspaces != SUPPORTED_WANDB_WORKSPACES_VERSION:
        raise MissingWandbReportExtraError(
            "W&B Research Report publication requires "
            f"wandb-workspaces=={SUPPORTED_WANDB_WORKSPACES_VERSION}; found "
            f"{observed_workspaces}"
        )
    api = getattr(wandb, "Api", None)
    report = getattr(reports, "Report", None)
    markdown_block = getattr(reports, "MarkdownBlock", None)
    if not callable(api) or not callable(report) or not callable(markdown_block):
        raise MissingWandbReportExtraError(
            "the installed W&B packages do not expose the required Reports v2 API"
        )
    if not callable(getattr(report, "save", None)) or not callable(
        getattr(report, "from_url", None)
    ):
        raise MissingWandbReportExtraError(
            "the installed W&B Reports v2 API lacks save or authoritative readback"
        )
    try:
        readback_parameters = inspect.signature(report.from_url).parameters
    except (TypeError, ValueError) as exc:
        raise MissingWandbReportExtraError(
            "the installed W&B Reports v2 API cannot prove raw-model readback support"
        ) from exc
    if "as_model" not in readback_parameters:
        raise MissingWandbReportExtraError(
            "the installed W&B Reports v2 API lacks raw-model readback support"
        )


def _publish_with_wandb_report_api(
    *,
    wandb: Any,
    reports: Any,
    projection: ResearchIndexReportProjectionV1,
    publication_env: Mapping[str, str],
    expected_report_id: str | None = None,
    expected_report_url: str | None = None,
) -> ResearchIndexReportPublicationOutcomeV1:
    _verify_projection_compatibility(projection)
    entity, project_id = _project_parts(projection.target.project)
    marker = _projection_marker(projection.projection_digest)
    title = _report_title(projection)
    description = _report_description(projection, marker=marker)
    markdown = _report_markdown(projection, marker=marker)
    scoped_environment = {
        **publication_env,
        "WANDB_BASE_URL": projection.target.api_base_url,
        "WANDB_APP_BASE_URL": projection.target.app_base_url,
        "WANDB_APP_URL": projection.target.app_base_url,
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project_id,
    }
    if (expected_report_id is None) != (expected_report_url is None):
        raise WandbResearchReportPublicationError(
            "existing W&B Report verification requires both id and URL"
        )

    try:
        with _PUBLICATION_LOCK, _scoped_environment(scoped_environment):
            api = wandb.Api()
            if expected_report_id is not None and expected_report_url is not None:
                verified_url = _validated_report_url(
                    expected_report_url,
                    target=projection.target,
                    entity=entity,
                    project_id=project_id,
                    report_id=expected_report_id,
                )
                outcome = _read_verified_report(
                    reports=reports,
                    projection=projection,
                    report_url=verified_url,
                    expected_report_id=expected_report_id,
                    expected_title=title,
                    expected_description=description,
                    expected_markdown=markdown,
                )
                _verify_unique_marker(
                    api,
                    project=projection.target.project,
                    marker=marker,
                    expected_report_id=expected_report_id,
                    expected_title=title,
                )
                return outcome
            matches = _find_marker_reports(
                api,
                project=projection.target.project,
                marker=marker,
                expected_title=title,
            )
            if len(matches) > 1:
                raise WandbResearchReportPublicationError(
                    "W&B contains more than one Report with the exact projection marker"
                )
            if matches:
                candidate = matches[0]
                candidate_id = str(getattr(candidate, "id", "") or "")
                candidate_url = _validated_report_url(
                    getattr(candidate, "url", None),
                    target=projection.target,
                    entity=entity,
                    project_id=project_id,
                    report_id=candidate_id,
                )
                outcome = _read_verified_report(
                    reports=reports,
                    projection=projection,
                    report_url=candidate_url,
                    expected_report_id=candidate_id,
                    expected_title=title,
                    expected_description=description,
                    expected_markdown=markdown,
                )
                _verify_unique_marker(
                    wandb.Api(),
                    project=projection.target.project,
                    marker=marker,
                    expected_report_id=candidate_id,
                    expected_title=title,
                )
                return outcome

            authored = reports.Report(
                entity=entity,
                project=project_id,
                title=title,
                description=description,
                width="readable",
                blocks=[reports.MarkdownBlock(text=markdown)],
            )
            saved = authored.save(draft=False, clone=False)
            if saved is None:
                raise RetryableWandbResearchReportPublicationError(
                    "W&B Reports v2 save returned no authored Report"
                )
            report_id = str(getattr(saved, "id", "") or "")
            report_url = _validated_report_url(
                getattr(saved, "url", None),
                target=projection.target,
                entity=entity,
                project_id=project_id,
                report_id=report_id,
            )
            outcome = _read_verified_report(
                reports=reports,
                projection=projection,
                report_url=report_url,
                expected_report_id=report_id,
                expected_title=title,
                expected_description=description,
                expected_markdown=markdown,
            )
            _verify_unique_marker(
                wandb.Api(),
                project=projection.target.project,
                marker=marker,
                expected_report_id=report_id,
                expected_title=title,
            )
            return outcome
    except WandbResearchReportPublicationError:
        raise
    except Exception as exc:
        raise _classified_remote_error(
            "W&B Research Report publication or authoritative readback failed",
            exc,
        ) from exc


def _read_verified_report(
    *,
    reports: Any,
    projection: ResearchIndexReportProjectionV1,
    report_url: str,
    expected_report_id: str,
    expected_title: str,
    expected_description: str,
    expected_markdown: str,
) -> ResearchIndexReportPublicationOutcomeV1:
    try:
        report = reports.Report.from_url(report_url)
        raw_model = reports.Report.from_url(report_url, as_model=True)
    except Exception as exc:
        raise _classified_remote_error(
            "W&B Research Report authoritative readback failed",
            exc,
        ) from exc
    return _verified_outcome(
        reports=reports,
        projection=projection,
        report=report,
        raw_model=raw_model,
        expected_report_id=expected_report_id,
        expected_title=expected_title,
        expected_description=expected_description,
        expected_markdown=expected_markdown,
    )


def _verified_outcome(
    *,
    reports: Any,
    projection: ResearchIndexReportProjectionV1,
    report: Any,
    raw_model: Any,
    expected_report_id: str,
    expected_title: str,
    expected_description: str,
    expected_markdown: str,
) -> ResearchIndexReportPublicationOutcomeV1:
    entity, project_id = _project_parts(projection.target.project)
    report_id = str(getattr(report, "id", "") or "")
    if not report_id or report_id != expected_report_id:
        raise WandbResearchReportPublicationError(
            "the saved W&B Report id does not match the expected Report id"
        )
    if (
        str(getattr(report, "entity", "") or "") != entity
        or str(getattr(report, "project", "") or "") != project_id
    ):
        raise WandbResearchReportPublicationError(
            "the saved W&B Report project does not match the expected project"
        )
    if str(getattr(report, "title", "") or "") != expected_title:
        raise WandbResearchReportPublicationError(
            "the saved W&B Report title does not match the expected title"
        )
    if str(getattr(report, "description", "") or "") != expected_description:
        raise WandbResearchReportPublicationError(
            "the saved W&B Report description does not match the expected description"
        )
    blocks = list(getattr(report, "blocks", ()) or ())
    markdown_type = reports.MarkdownBlock
    if len(blocks) != 1 or not isinstance(blocks[0], markdown_type):
        raise WandbResearchReportPublicationError(
            "the saved W&B Report blocks do not match the expected blocks"
        )
    if str(getattr(blocks[0], "text", "") or "") != expected_markdown:
        raise WandbResearchReportPublicationError(
            "the saved W&B Report Markdown does not match the expected Markdown"
        )
    marker = _projection_marker(projection.projection_digest)
    if marker not in expected_description or marker not in expected_markdown:
        raise WandbResearchReportPublicationError(
            "W&B Report projection marker is missing from controlled content"
        )
    _verify_raw_report_model(
        raw_model,
        expected_report_id=expected_report_id,
        expected_entity=entity,
        expected_project=project_id,
        expected_title=expected_title,
        expected_description=expected_description,
        expected_markdown=expected_markdown,
    )
    report_url = _validated_report_url(
        getattr(report, "url", None),
        target=projection.target,
        entity=entity,
        project_id=project_id,
        report_id=report_id,
    )
    return ResearchIndexReportPublicationOutcomeV1(
        target=projection.target,
        report_id=report_id,
        report_url=report_url,
        report_api=WANDB_REPORT_API,
        report_api_version=SUPPORTED_WANDB_WORKSPACES_VERSION,
        api_stability=WANDB_REPORT_API_STABILITY,
        readback_projection_digest=projection.projection_digest,
        rendered_content_digest=_rendered_content_digest(
            title=expected_title,
            description=expected_description,
            markdown=expected_markdown,
        ),
        readback_status="reconciled",
        publisher_id=WANDB_RESEARCH_REPORT_PUBLISHER_ID,
        publisher_revision=WANDB_RESEARCH_REPORT_PUBLISHER_REVISION,
    )


def _verify_raw_report_model(
    model: Any,
    *,
    expected_report_id: str,
    expected_entity: str,
    expected_project: str,
    expected_title: str,
    expected_description: str,
    expected_markdown: str,
) -> None:
    project = getattr(model, "project", None)
    spec = getattr(model, "spec", None)
    if (
        str(getattr(model, "id", "") or "") != expected_report_id
        or str(getattr(model, "display_name", "") or "") != expected_title
        or str(getattr(model, "description", "") or "") != expected_description
        or str(getattr(project, "entity_name", "") or "") != expected_entity
        or str(getattr(project, "name", "") or "") != expected_project
        or str(getattr(spec, "width", "") or "") != "readable"
    ):
        raise WandbResearchReportPublicationError(
            "the saved W&B Report fields do not match the expected fields"
        )
    raw_blocks = list(getattr(spec, "blocks", ()) or ())
    if (
        len(raw_blocks) != 3
        or type(raw_blocks[0]).__name__ != "Paragraph"
        or type(raw_blocks[1]).__name__ != "MarkdownBlock"
        or type(raw_blocks[2]).__name__ != "Paragraph"
        or str(getattr(raw_blocks[1], "content", "") or "") != expected_markdown
    ):
        raise WandbResearchReportPublicationError(
            "the saved W&B Report model blocks do not match the expected blocks"
        )


def _find_marker_reports(
    api: Any,
    *,
    project: str,
    marker: str,
    expected_title: str,
    fail_on_title_conflict: bool = True,
) -> list[Any]:
    try:
        candidates = api.reports(project, per_page=50)
    except Exception as exc:
        raise _classified_remote_error(
            "W&B could not list Reports in the exact project",
            exc,
        ) from exc
    matches: list[Any] = []
    for position, candidate in enumerate(candidates):
        if position >= _MAX_REPORT_SCAN:
            raise WandbResearchReportPublicationError(
                "W&B Report scan exceeded the bounded publication limit"
            )
        searchable = _report_searchable_text(candidate)
        if marker in searchable:
            matches.append(candidate)
        elif (
            fail_on_title_conflict and _report_display_name(candidate) == expected_title
        ):
            raise WandbResearchReportPublicationError(
                "W&B contains the digest-addressed Report title without its exact "
                "projection marker"
            )
    return matches


def _verify_unique_marker(
    api: Any,
    *,
    project: str,
    marker: str,
    expected_report_id: str,
    expected_title: str,
) -> None:
    matches = _find_marker_reports(
        api,
        project=project,
        marker=marker,
        expected_title=expected_title,
        fail_on_title_conflict=False,
    )
    if not matches:
        raise RetryableWandbResearchReportPublicationError(
            "the saved W&B Report is not yet visible in authoritative listing"
        )
    if len(matches) > 1:
        raise WandbResearchReportPublicationError(
            "W&B contains more than one Report with the exact projection marker"
        )
    observed_id = str(getattr(matches[0], "id", "") or "")
    if observed_id != expected_report_id:
        raise WandbResearchReportPublicationError(
            "the W&B Report marker resolves to another Report id"
        )


def _classified_remote_error(
    message: str,
    error: Exception,
) -> WandbResearchReportPublicationError:
    chain = tuple(_error_chain(error))
    if any(isinstance(item, PermissionError) for item in chain):
        return WandbResearchReportPublicationError(message)
    names = tuple(type(item).__name__.casefold() for item in chain)
    permanent_markers = (
        "auth",
        "permission",
        "forbidden",
        "unauthorized",
        "validation",
        "usage",
        "schema",
    )
    if any(marker in name for name in names for marker in permanent_markers):
        return WandbResearchReportPublicationError(message)
    statuses = tuple(
        status for item in chain if (status := _http_status_code(item)) is not None
    )
    if any(
        400 <= status < 500 and status not in {408, 425, 429} for status in statuses
    ):
        return WandbResearchReportPublicationError(message)
    if any(status in {408, 425, 429} or status >= 500 for status in statuses):
        return RetryableWandbResearchReportPublicationError(message)
    retryable_markers = (
        "commerror",
        "timeout",
        "connection",
        "connect",
        "temporary",
        "unavailable",
        "serverbusy",
        "ratelimit",
    )
    if any(
        isinstance(item, (TimeoutError, ConnectionError, OSError)) for item in chain
    ) or any(marker in name for name in names for marker in retryable_markers):
        return RetryableWandbResearchReportPublicationError(message)
    return WandbResearchReportPublicationError(message)


def _error_chain(error: Exception) -> Iterator[Exception]:
    pending = [error]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        for attribute in ("exc", "__cause__", "__context__"):
            nested = getattr(item, attribute, None)
            if isinstance(nested, Exception):
                pending.append(nested)


def _http_status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    return None


def _report_searchable_text(report: Any) -> str:
    parts = [
        str(getattr(report, "display_name", "") or ""),
        str(getattr(report, "title", "") or ""),
        str(getattr(report, "description", "") or ""),
    ]
    spec = getattr(report, "spec", None)
    if spec is not None:
        try:
            parts.append(json.dumps(spec, sort_keys=True, default=str))
        except (TypeError, ValueError):
            parts.append(str(spec))
    return "\n".join(parts)


def _report_display_name(report: Any) -> str:
    return str(
        getattr(report, "display_name", "") or getattr(report, "title", "") or ""
    )


def _verify_projection_compatibility(
    projection: ResearchIndexReportProjectionV1,
) -> None:
    expected = {
        "renderer_id": WANDB_RESEARCH_REPORT_PUBLISHER_ID,
        "renderer_revision": "v1",
        "report_api": WANDB_REPORT_API,
        "report_api_version": SUPPORTED_WANDB_WORKSPACES_VERSION,
        "api_stability": WANDB_REPORT_API_STABILITY,
        "report_width": "readable",
    }
    for field, value in expected.items():
        if str(getattr(projection, field, "") or "") != value:
            raise WandbResearchReportPublicationError(
                f"W&B Report projection has an unsupported {field}"
            )
    if not _DIGEST.fullmatch(str(projection.projection_digest)):
        raise WandbResearchReportPublicationError(
            "W&B Report projection digest is not canonical SHA-256"
        )
    if int(projection.study_count) < 1 or int(projection.total_rows) < 1:
        raise WandbResearchReportPublicationError(
            "W&B Report projection must contain nonzero Studies and rows"
        )
    if len(tuple(projection.studies)) != int(projection.study_count):
        raise WandbResearchReportPublicationError(
            "W&B Report projection Study count disagrees"
        )


def _report_title(projection: ResearchIndexReportProjectionV1) -> str:
    title = str(projection.report_title).strip()
    suffix = projection.projection_digest[:12]
    if not title or len(title) > 128 or not title.endswith(suffix):
        raise WandbResearchReportPublicationError(
            "W&B Report title must contain 128 characters or fewer and end with "
            "the first 12 characters of the projection digest"
        )
    return title


def _report_description(
    projection: ResearchIndexReportProjectionV1, *, marker: str
) -> str:
    description = str(projection.report_description).strip()
    if marker not in description:
        description = f"{description}\n\n{marker}" if description else marker
    return description


def _report_markdown(
    projection: ResearchIndexReportProjectionV1, *, marker: str
) -> str:
    lines = [
        "## Fugue Research index",
        "",
        "**API status: Public Preview.** This W&B Report is an optional "
        "presentation of a Fugue Research index. It lists summary fields from "
        "immutable Fugue ComparisonResultV3 artifacts. It does not create or "
        "replace first-class Study records.",
        "",
        "The W&B project and Report settings control access. Fugue does not request "
        "a public share link.",
        "",
        marker,
        "",
        f"- Research ID: `{_inline_code(projection.research_id)}`",
        f"- Index digest: `{projection.index_digest}`",
        f"- Studies: {projection.study_count}",
        f"- Result rows: {projection.total_rows}",
        f"- [Index Run]({_safe_bound_url(projection.index_run_url)})",
        f"- [Immutable index Artifact version]({_safe_bound_url(projection.index_artifact_url)})",
        "",
        "| Study | Comparison | Behavioral finding | Behavioral next action | Governed decision | Decision next action | Task validity | Result rows | Candidates | Evidence integrity (not task quality) | Links |",
        "|---|---|---|---|---|---|---|---:|---|---|---|",
    ]
    for study in sorted(projection.studies, key=lambda item: item.study_id):
        evidence = (
            f"grade {_table_text(study.evidence_integrity_grade)} for evidence "
            "links and privacy checks; result backend: "
            f"{_table_text(study.evidence_backend)}; local chain: "
            f"{_table_text(study.local_chain_integrity)}; W&B publication: "
            f"{_table_text(study.published_chain_integrity)}"
        )
        links = (
            f"[Weave project]({_safe_bound_url(study.evidence_project_url)}) · "
            "[prediction-and-score call]"
            f"({_safe_bound_url(study.primary_evidence_url)})"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _table_text(study.study_id),
                    _table_text(study.comparison_id),
                    _table_text(study.behavioral_status),
                    _table_text(study.behavioral_recommendation),
                    _table_text(study.decision_status),
                    _table_text(study.decision_recommendation),
                    _table_text(study.task_validity_status),
                    str(study.rows),
                    _candidate_assignments(study.candidate_assignments),
                    evidence,
                    links,
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Deterministic outcomes, advisory judgments, mechanism evidence, "
            "efficiency, and evidence integrity remain separate claims.",
            "",
        )
    )
    return "\n".join(lines)


def _projection_marker(digest: str) -> str:
    if not _DIGEST.fullmatch(str(digest)):
        raise WandbResearchReportPublicationError(
            "W&B Report projection digest is not canonical SHA-256"
        )
    return f"{_MARKER_PREFIX}{digest}"


def _rendered_content_digest(
    *,
    title: str,
    description: str,
    markdown: str,
) -> str:
    return stable_digest(
        {
            "schema_version": 1,
            "title": title,
            "description": description,
            "width": "readable",
            "blocks": [{"type": "MarkdownBlock", "text": markdown}],
        }
    )


def _candidate_assignments(assignments: Any) -> str:
    rendered = []
    for assignment in sorted(
        tuple(assignments),
        key=lambda item: (item.role, item.harness, item.candidate_id),
    ):
        rendered.append(
            _table_text(
                f"{assignment.role}:{assignment.harness}@{assignment.candidate_id}"
            )
        )
    if not rendered:
        raise WandbResearchReportPublicationError(
            "W&B Report Study has no candidate assignments"
        )
    return "; ".join(rendered)


def _safe_bound_url(raw: Any) -> str:
    value = str(raw or "")
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or any(character.isspace() for character in value)
        or any(unicodedata.category(character) == "Cc" for character in value)
        or any(character in "\\[]()<>'\"`" for character in value)
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise WandbResearchReportPublicationError(
            "W&B Report projection contains an unsafe link"
        )
    return value


def _table_text(raw: Any) -> str:
    value = str(raw).replace("\r", " ").replace("\n", " ").strip()
    escaped = []
    for character in value:
        if character in "\\|`*_{}[]<>()#+!~":
            escaped.append("\\")
        escaped.append(character)
    return "".join(escaped)


def _inline_code(raw: Any) -> str:
    return str(raw).replace("`", "\\`").replace("\r", " ").replace("\n", " ")


def _project_parts(project: str) -> tuple[str, str]:
    parts = str(project).split("/")
    if len(parts) != 2 or not all(_PROJECT_PART.fullmatch(part) for part in parts):
        raise WandbResearchReportPublicationError(
            "W&B Report project must be ENTITY/PROJECT"
        )
    return parts[0], parts[1]


def _validated_report_url(
    raw: Any,
    *,
    target: Any,
    entity: str,
    project_id: str,
    report_id: str,
) -> str:
    value = str(raw or "")
    parsed = urlsplit(value)
    base = urlsplit(str(target.app_base_url))
    if (
        value != value.strip()
        or any(character.isspace() for character in value)
        or any(unicodedata.category(character) == "Cc" for character in value)
        or any(character in "\\[]()<>'\"`" for character in value)
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise WandbResearchReportPublicationError(
            "W&B Report URL is not a safe canonical HTTPS URL"
        )
    if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc):
        raise WandbResearchReportPublicationError(
            "W&B Report URL returned another W&B application origin"
        )
    if not report_id or "/" in report_id:
        raise WandbResearchReportPublicationError("W&B Report has no safe id")
    path = [part for part in parsed.path.split("/") if part]
    accepted_id_suffixes = {f"--{report_id}", f"--{report_id.rstrip('=')}"}
    if (
        len(path) != 4
        or path[:3] != [entity, project_id, "reports"]
        or not any(path[3].endswith(suffix) for suffix in accepted_id_suffixes)
        or path[3] in accepted_id_suffixes
    ):
        raise WandbResearchReportPublicationError(
            "W&B Report URL returned another Report object path"
        )
    return value


@contextmanager
def _scoped_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _SCOPED_ENV}
    try:
        for name in _SCOPED_ENV:
            if name in values:
                os.environ[name] = str(values[name])
            else:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
