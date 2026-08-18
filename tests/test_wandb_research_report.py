from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest
from requests import Response
from requests.exceptions import HTTPError
from wandb.errors import CommError as WandbCommError

import fugue.bench.wandb_research_report as adapter
from fugue import weave_support
from fugue.bench.research_index import ResearchIndexPublicationTargetV1


def _digest(character: str) -> str:
    return character * 64


@dataclass(frozen=True)
class _Projection:
    research_id: str
    index_digest: str
    index_file_sha256: str
    index_publication_id: str
    index_publication_receipt_digest: str
    index_publication_receipt_file_sha256: str
    target: ResearchIndexPublicationTargetV1
    index_run_url: str
    index_artifact_url: str
    renderer_id: str
    renderer_revision: str
    report_api: str
    report_api_version: str
    api_stability: str
    report_title: str
    report_description: str
    report_width: str
    study_count: int
    total_rows: int
    studies: tuple[Any, ...]
    projection_digest: str


def _target() -> ResearchIndexPublicationTargetV1:
    return ResearchIndexPublicationTargetV1(
        project="wandb/community-studies",
        api_base_url="https://api.wandb.example",
        app_base_url="https://app.wandb.example",
    )


def _study(study_id: str, *, rows: int, status: str) -> SimpleNamespace:
    return SimpleNamespace(
        study_id=study_id,
        comparison_id=f"comparison-{study_id}",
        behavioral_status=status,
        behavioral_recommendation=(
            "Advance to confirmation." if status == "improved" else "Repair tasks."
        ),
        decision_status="inconclusive",
        decision_recommendation="Package release was not evaluated.",
        task_validity_status="valid",
        rows=rows,
        evidence_integrity_grade="A",
        evidence_backend="local",
        local_chain_integrity="reconciled",
        published_chain_integrity="reconciled",
        candidate_assignments=(
            SimpleNamespace(
                role="baseline",
                harness="claude-code",
                candidate_id=_digest("b"),
            ),
            SimpleNamespace(
                role="candidate",
                harness="claude-code",
                candidate_id=_digest("c"),
            ),
        ),
        evidence_project_url=(f"https://app.wandb.example/wandb/{study_id}/workspace"),
        primary_evidence_url=(
            f"https://app.wandb.example/wandb/{study_id}/weave/calls/call-1"
        ),
    )


def _projection(seed: str = "a") -> _Projection:
    projection_digest = _digest(seed)
    return _Projection(
        research_id="community-skill-studies-v1",
        index_digest=_digest("1"),
        index_file_sha256=_digest("2"),
        index_publication_id=_digest("3"),
        index_publication_receipt_digest=_digest("4"),
        index_publication_receipt_file_sha256=_digest("5"),
        target=_target(),
        index_run_url=(
            "https://app.wandb.example/wandb/community-studies/runs/index-run"
        ),
        index_artifact_url=(
            "https://app.wandb.example/wandb/community-studies/artifacts/"
            "study-index/research-index/v1"
        ),
        renderer_id=adapter.WANDB_RESEARCH_REPORT_PUBLISHER_ID,
        renderer_revision="v1",
        report_api=adapter.WANDB_REPORT_API,
        report_api_version=adapter.SUPPORTED_WANDB_WORKSPACES_VERSION,
        api_stability="public_preview",
        report_title=f"Community Skill studies · {projection_digest[:12]}",
        report_description="Decision-ready projections for two locked Studies.",
        report_width="readable",
        study_count=2,
        total_rows=12,
        studies=(
            _study("study-b", rows=8, status="unchanged"),
            _study("study-a", rows=4, status="improved"),
        ),
        projection_digest=projection_digest,
    )


class _FakeMarkdownBlock:
    def __init__(self, *, text: str) -> None:
        self.text = text


class Paragraph:
    """Match the sentinel block name returned by wandb-workspaces."""


class MarkdownBlock:
    """Match the raw block name and content field returned by the API model."""

    def __init__(self, *, content: str) -> None:
        self.content = content


@dataclass
class _ListedReport:
    id: str
    display_name: str
    description: str
    spec: dict[str, Any]
    url: str


class _ReadbackReport:
    def __init__(
        self,
        *,
        report_id: str,
        entity: str,
        project: str,
        title: str,
        description: str,
        width: str,
        blocks: list[_FakeMarkdownBlock],
        url: str,
    ) -> None:
        self.id = report_id
        self.entity = entity
        self.project = project
        self.title = title
        self.description = description
        self.width = width
        self.blocks = blocks
        self.url = url

    def enable_share_link(self) -> None:
        raise AssertionError("publisher must not enable a public share link")

    def get_share_url(self) -> str:
        raise AssertionError("publisher must not request a public share link")


@dataclass
class _RawProject:
    entity_name: str
    name: str


@dataclass
class _RawSpec:
    width: str
    blocks: list[Any]


@dataclass
class _RawReportModel:
    id: str
    display_name: str
    description: str
    project: _RawProject
    spec: _RawSpec


class _AuthoredReport:
    def __init__(self, sdk: _FakeWandb, **kwargs: Any) -> None:
        self._sdk = sdk
        self.id = ""
        self.entity = kwargs["entity"]
        self.project = kwargs["project"]
        self.title = kwargs["title"]
        self.description = kwargs["description"]
        self.width = kwargs["width"]
        self.blocks = list(kwargs["blocks"])
        self.url = ""

    def save(self, *, draft: bool, clone: bool) -> _AuthoredReport | None:
        self._sdk.save_calls.append((self, draft, clone, dict(os.environ)))
        if self._sdk.save_exception is not None:
            raise self._sdk.save_exception
        self.id = self._sdk.report_id_override or (
            f"report-{len(self._sdk.listed_reports) + 1}"
        )
        url_id = self.id.rstrip("=") if self._sdk.padding_free_url else self.id
        slug = self.title.replace(" ", "-")
        self.url = (
            f"{self._sdk.app_base_url}/{self.entity}/{self.project}/reports/"
            f"{slug}--{url_id}"
        )
        readback = _ReadbackReport(
            report_id=self.id,
            entity=self.entity,
            project=self.project,
            title=self.title,
            description=self.description,
            width=self.width,
            blocks=[_FakeMarkdownBlock(text=block.text) for block in self.blocks],
            url=self.url,
        )
        self._sdk.readbacks[self.url] = readback
        self._sdk.listed_reports.append(
            _ListedReport(
                id=self.id,
                display_name=self.title,
                description=self.description,
                spec={"blocks": [{"text": self.blocks[0].text}]},
                url=self.url,
            )
        )
        if self._sdk.duplicate_on_save:
            self._sdk.listed_reports.append(
                replace(
                    self._sdk.listed_reports[-1],
                    id=f"duplicate-{self.id}",
                )
            )
        return None if self._sdk.save_returns_none else self

    def enable_share_link(self) -> None:
        self._sdk.share_calls.append("enable")

    def get_share_url(self) -> str:
        self._sdk.share_calls.append("get")
        return "https://unsafe.example/accessToken=secret"


class _FakeReportFactory:
    sdk: _FakeWandb

    def __new__(cls, **kwargs: Any) -> _AuthoredReport:
        return _AuthoredReport(cls.sdk, **kwargs)

    def save(self, *, draft: bool = False, clone: bool = False):
        raise AssertionError("feature-check placeholder must not run")

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        as_model: bool = False,
    ) -> _ReadbackReport | _RawReportModel:
        cls.sdk.from_url_calls.append((url, as_model, dict(os.environ)))
        if cls.sdk.duplicate_on_readback:
            cls.sdk.duplicate_on_readback = False
            cls.sdk.listed_reports.append(
                replace(
                    cls.sdk.listed_reports[-1],
                    id=f"duplicate-{cls.sdk.listed_reports[-1].id}",
                )
            )
        if cls.sdk.readback_exception is not None:
            raise cls.sdk.readback_exception
        report = cls.sdk.readbacks[url]
        if as_model:
            return cls.sdk.mutate_raw_model(report)
        return cls.sdk.mutate_readback(report)


class _FakeApi:
    def __init__(self, sdk: _FakeWandb) -> None:
        self._sdk = sdk

    def reports(self, project: str, *, per_page: int):
        self._sdk.report_queries.append((project, per_page, dict(os.environ)))
        if self._sdk.list_exception is not None:
            raise self._sdk.list_exception
        return list(self._sdk.listed_reports)


class _FakeWandb:
    __version__ = adapter.SUPPORTED_WANDB_VERSION

    def __init__(self) -> None:
        self.app_base_url = "https://app.wandb.example"
        self.api_calls: list[dict[str, str]] = []
        self.report_queries: list[tuple[str, int, dict[str, str]]] = []
        self.save_calls: list[tuple[_AuthoredReport, bool, bool, dict[str, str]]] = []
        self.from_url_calls: list[tuple[str, bool, dict[str, str]]] = []
        self.listed_reports: list[_ListedReport] = []
        self.readbacks: dict[str, _ReadbackReport] = {}
        self.list_exception: Exception | None = None
        self.save_exception: Exception | None = None
        self.readback_exception: Exception | None = None
        self.save_returns_none = False
        self.readback_mutation: str | None = None
        self.raw_model_mutation: str | None = None
        self.report_id_override: str | None = None
        self.padding_free_url = False
        self.duplicate_on_save = False
        self.duplicate_on_readback = False
        self.share_calls: list[str] = []

    def Api(self) -> _FakeApi:
        self.api_calls.append(dict(os.environ))
        return _FakeApi(self)

    def mutate_readback(self, report: _ReadbackReport) -> _ReadbackReport:
        observed = _ReadbackReport(
            report_id=report.id,
            entity=report.entity,
            project=report.project,
            title=report.title,
            description=report.description,
            width=report.width,
            blocks=[_FakeMarkdownBlock(text=block.text) for block in report.blocks],
            url=report.url,
        )
        mutation = self.readback_mutation
        if mutation == "id":
            observed.id = "another-id"
        elif mutation == "entity":
            observed.entity = "another-entity"
        elif mutation == "project":
            observed.project = "another-project"
        elif mutation == "title":
            observed.title = "Manually changed"
        elif mutation == "description":
            observed.description += " changed"
        elif mutation == "width":
            observed.width = "fluid"
        elif mutation == "blocks":
            observed.blocks.append(_FakeMarkdownBlock(text="unexpected"))
        elif mutation == "content":
            observed.blocks[0].text += " changed"
        elif mutation == "url":
            observed.url = observed.url.replace(
                "/community-studies/", "/another-project/"
            )
        return observed

    def mutate_raw_model(self, report: _ReadbackReport) -> _RawReportModel:
        model = _RawReportModel(
            id=report.id,
            display_name=report.title,
            description=report.description,
            project=_RawProject(
                entity_name=report.entity,
                name=report.project,
            ),
            spec=_RawSpec(
                width=report.width,
                blocks=[
                    Paragraph(),
                    MarkdownBlock(content=report.blocks[0].text),
                    Paragraph(),
                ],
            ),
        )
        mutation = self.raw_model_mutation or self.readback_mutation
        if mutation == "raw_id":
            model.id = "another-id"
        elif mutation == "raw_title":
            model.display_name = "Manually changed"
        elif mutation == "raw_description":
            model.description += " changed"
        elif mutation == "raw_entity":
            model.project.entity_name = "another-entity"
        elif mutation == "raw_project":
            model.project.name = "another-project"
        elif mutation == "width":
            model.spec.width = "fluid"
        elif mutation == "raw_blocks":
            model.spec.blocks.append(Paragraph())
        elif mutation == "raw_content":
            model.spec.blocks[1].content += " changed"
        return model


def _reports_module(sdk: _FakeWandb) -> SimpleNamespace:
    _FakeReportFactory.sdk = sdk
    return SimpleNamespace(
        Report=_FakeReportFactory,
        MarkdownBlock=_FakeMarkdownBlock,
    )


def _publisher(
    monkeypatch: pytest.MonkeyPatch,
    sdk: _FakeWandb,
    *,
    env: dict[str, str] | None = None,
):
    reports = _reports_module(sdk)

    def import_module(name: str):
        if name == "wandb":
            return sdk
        if name == "wandb_workspaces.reports.v2":
            return reports
        raise AssertionError(name)

    monkeypatch.setattr(adapter, "ResearchIndexReportProjectionV1", _Projection)
    monkeypatch.setattr(adapter.importlib, "import_module", import_module)
    monkeypatch.setattr(
        adapter.importlib.metadata,
        "version",
        lambda name: (
            adapter.SUPPORTED_WANDB_WORKSPACES_VERSION
            if name == "wandb-workspaces"
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )
    return adapter.wandb_research_report_publisher_from_environment(
        env
        or {
            "WANDB_API_KEY": "publication-key",
            "WANDB_BASE_URL": "https://api.wandb.example",
            "WANDB_APP_BASE_URL": "https://app.wandb.example",
        }
    )


def test_optional_import_is_lazy_and_missing_extra_fails_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    real_import = importlib.import_module

    def spy(name: str, package: str | None = None):
        calls.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", spy)
    importlib.reload(adapter)
    assert "wandb" not in calls
    assert "wandb_workspaces.reports.v2" not in calls

    def missing(name: str):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(adapter.importlib, "import_module", missing)
    with pytest.raises(
        adapter.MissingWandbReportExtraError,
        match=r"fugue\[wandb-report\]",
    ):
        adapter.wandb_research_report_publisher_from_environment({})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WANDB_BASE_URL", "https://credential-exfiltration.example"),
        ("WANDB_APP_BASE_URL", "https://another-app.example"),
    ],
)
def test_destination_mismatch_blocks_before_api_construction(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    sdk = _FakeWandb()
    env = {
        "WANDB_API_KEY": "publication-key",
        "WANDB_BASE_URL": "https://api.wandb.example",
        "WANDB_APP_BASE_URL": "https://app.wandb.example",
        name: value,
    }
    publisher = _publisher(monkeypatch, sdk, env=env)

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="target does not match",
    ):
        publisher(_projection())

    assert sdk.api_calls == []


def test_invalid_destination_environment_is_a_typed_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(
        monkeypatch,
        sdk,
        env={
            "WANDB_API_KEY": "publication-key",
            "WANDB_BASE_URL": "http://insecure.example",
            "WANDB_APP_BASE_URL": "https://app.wandb.example",
        },
    )

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="environment is invalid",
    ):
        publisher(_projection())

    assert sdk.api_calls == []


def test_partial_existing_report_identity_blocks_before_api_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="requires both id and URL",
    ):
        publisher(_projection(), expected_report_id="VmlldzoxMjM=")

    assert sdk.api_calls == []


@pytest.mark.parametrize("control", ["\x00", "\x1b", "\x7f"])
def test_renderer_rejects_control_bytes_in_bound_urls(control: str) -> None:
    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="unsafe link",
    ):
        adapter._safe_bound_url(
            f"https://wandb.ai/wandb/research-index/runs/run{control}one"
        )


@pytest.mark.parametrize("dependency", ["wandb", "wandb-workspaces"])
def test_pinned_version_mismatch_blocks_before_api_client(
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    sdk = _FakeWandb()
    if dependency == "wandb":
        sdk.__version__ = "0.29.0"
    reports = _reports_module(sdk)
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: sdk if name == "wandb" else reports,
    )
    monkeypatch.setattr(
        adapter.importlib.metadata,
        "version",
        lambda _name: (
            "0.5.0"
            if dependency == "wandb-workspaces"
            else adapter.SUPPORTED_WANDB_WORKSPACES_VERSION
        ),
    )

    with pytest.raises(adapter.MissingWandbReportExtraError, match=dependency):
        adapter.wandb_research_report_publisher_from_environment({})
    assert sdk.api_calls == []


def test_create_uses_one_controlled_block_and_authoritative_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    previous = {
        name: f"old-{position}" for position, name in enumerate(adapter._SCOPED_ENV)
    }
    for name, value in previous.items():
        monkeypatch.setenv(name, value)
    projection = _projection()

    outcome = _publisher(monkeypatch, sdk)(projection)

    assert len(sdk.save_calls) == 1
    authored, draft, clone, observed_env = sdk.save_calls[0]
    assert type(authored) is _AuthoredReport
    assert draft is False
    assert clone is False
    assert authored.entity == "wandb"
    assert authored.project == "community-studies"
    assert authored.width == "readable"
    assert len(authored.blocks) == 1
    markdown = authored.blocks[0].text
    marker = adapter._projection_marker(projection.projection_digest)
    assert marker in authored.description
    assert marker in markdown
    assert "Public Preview" in markdown
    assert "optional presentation" in markdown
    assert "does not create or replace first-class Study records" in markdown
    assert "project and Report settings" in markdown
    assert markdown.index("study-a") < markdown.index("study-b")
    assert "Advance to confirmation." in markdown
    assert "Repair tasks." in markdown
    assert "Index Run" in markdown
    assert "Immutable index Artifact" in markdown
    assert sdk.share_calls == []
    assert observed_env["WANDB_ENTITY"] == "wandb"
    assert observed_env["WANDB_PROJECT"] == "community-studies"
    assert observed_env["WANDB_API_KEY"] == "publication-key"
    assert observed_env["WANDB_BASE_URL"] == projection.target.api_base_url
    assert observed_env["WANDB_APP_BASE_URL"] == projection.target.app_base_url
    assert observed_env["WANDB_APP_URL"] == projection.target.app_base_url
    assert "WANDB_RUN_ID" not in observed_env
    assert type(sdk.readbacks[authored.url]) is _ReadbackReport
    assert len(sdk.api_calls) == 2
    assert [call[1] for call in sdk.from_url_calls] == [False, True]
    assert {name: os.environ.get(name) for name in previous} == previous
    assert outcome.target == projection.target
    assert outcome.report_id == authored.id
    assert outcome.report_url == authored.url
    assert "accessToken" not in outcome.report_url
    assert outcome.report_api == adapter.WANDB_REPORT_API
    assert outcome.report_api_version == adapter.SUPPORTED_WANDB_WORKSPACES_VERSION
    assert outcome.api_stability == "public_preview"
    assert outcome.readback_projection_digest == projection.projection_digest
    assert outcome.readback_status == "reconciled"


def test_exact_rerun_reuses_report_without_another_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    projection = _projection()

    first = publisher(projection)
    second = publisher(projection)

    assert second == first
    assert len(sdk.save_calls) == 1
    assert len(sdk.report_queries) == 4
    assert len(sdk.from_url_calls) == 4


def test_new_projection_digest_creates_a_distinct_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)

    first = publisher(_projection("a"))
    second = publisher(_projection("b"))

    assert first.report_id != second.report_id
    assert len(sdk.save_calls) == 2
    assert len(sdk.listed_reports) == 2


def test_duplicate_exact_markers_fail_without_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    projection = _projection()
    publisher(projection)
    sdk.listed_reports.append(replace(sdk.listed_reports[0], id="duplicate"))

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="more than one Report",
    ):
        publisher(projection)
    assert len(sdk.save_calls) == 1


def test_post_create_uniqueness_rejects_concurrent_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    sdk.duplicate_on_save = True

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="more than one Report",
    ):
        _publisher(monkeypatch, sdk)(_projection())

    assert len(sdk.save_calls) == 1
    assert len(sdk.listed_reports) == 2
    assert len(sdk.report_queries) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "id",
        "entity",
        "project",
        "title",
        "description",
        "width",
        "blocks",
        "content",
        "url",
    ],
)
def test_authoritative_readback_drift_fails(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    sdk = _FakeWandb()
    sdk.readback_mutation = mutation

    with pytest.raises(adapter.WandbResearchReportPublicationError):
        _publisher(monkeypatch, sdk)(_projection())


def test_existing_manually_edited_report_fails_instead_of_overwriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    projection = _projection()
    publisher(projection)
    sdk.readback_mutation = "content"

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="expected Markdown",
    ):
        publisher(projection)
    assert len(sdk.save_calls) == 1


def test_marker_reuse_detects_duplicate_created_during_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    projection = _projection()
    publisher(projection)
    sdk.duplicate_on_readback = True

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="more than one Report",
    ):
        publisher(projection)

    assert len(sdk.save_calls) == 1


def test_existing_receipt_direct_readback_never_creates_when_marker_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    projection = _projection()
    outcome = publisher(projection)
    sdk.listed_reports[0].description = "marker removed manually"
    sdk.listed_reports[0].spec = {}

    with pytest.raises(
        adapter.RetryableWandbResearchReportPublicationError,
        match="not yet visible",
    ):
        publisher(
            projection,
            expected_report_id=outcome.report_id,
            expected_report_url=outcome.report_url,
        )

    assert len(sdk.save_calls) == 1
    assert len(sdk.from_url_calls) == 4


@pytest.mark.parametrize("phase", ["list", "save", "readback", "save_none"])
def test_remote_unavailability_is_typed_retryable_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    sdk = _FakeWandb()
    if phase == "list":
        sdk.list_exception = OSError("offline")
    elif phase == "save":
        sdk.save_exception = TimeoutError("save timed out")
    elif phase == "readback":
        sdk.readback_exception = TimeoutError("readback timed out")
    else:
        sdk.save_returns_none = True
    monkeypatch.setenv("WANDB_ENTITY", "original-entity")

    with pytest.raises(adapter.RetryableWandbResearchReportPublicationError):
        _publisher(monkeypatch, sdk)(_projection())
    assert os.environ["WANDB_ENTITY"] == "original-entity"


class CommError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class SchemaError(Exception):
    pass


@pytest.mark.parametrize("error", [CommError("offline"), TimeoutError("slow")])
def test_transport_failures_are_retryable(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    sdk = _FakeWandb()
    sdk.list_exception = error

    with pytest.raises(adapter.RetryableWandbResearchReportPublicationError):
        _publisher(monkeypatch, sdk)(_projection())


@pytest.mark.parametrize(
    "error",
    [AuthenticationError("bad key"), PermissionError("denied"), SchemaError("bad")],
)
def test_permanent_auth_permission_and_schema_failures_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    sdk = _FakeWandb()
    sdk.list_exception = error

    with pytest.raises(adapter.WandbResearchReportPublicationError) as caught:
        _publisher(monkeypatch, sdk)(_projection())

    assert not isinstance(
        caught.value,
        adapter.RetryableWandbResearchReportPublicationError,
    )


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(403, False), (404, False), (429, True), (503, True)],
)
def test_wrapped_wandb_http_status_controls_retryability(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    retryable: bool,
) -> None:
    response = Response()
    response.status_code = status_code
    wrapped = WandbCommError(
        "W&B request failed",
        exc=HTTPError(f"HTTP {status_code}", response=response),
    )
    sdk = _FakeWandb()
    sdk.list_exception = wrapped

    expected = (
        adapter.RetryableWandbResearchReportPublicationError
        if retryable
        else adapter.WandbResearchReportPublicationError
    )
    with pytest.raises(expected) as caught:
        _publisher(monkeypatch, sdk)(_projection())

    assert (
        isinstance(
            caught.value,
            adapter.RetryableWandbResearchReportPublicationError,
        )
        is retryable
    )


def test_raw_report_width_drift_fails_authoritative_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    sdk.raw_model_mutation = "width"

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="fields do not match the expected fields",
    ):
        _publisher(monkeypatch, sdk)(_projection())


def test_retry_after_post_save_readback_failure_reuses_partial_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    publisher = _publisher(monkeypatch, sdk)
    sdk.readback_exception = TimeoutError("not finalized")

    with pytest.raises(adapter.RetryableWandbResearchReportPublicationError):
        publisher(_projection())
    assert len(sdk.save_calls) == 1
    sdk.readback_exception = None

    outcome = publisher(_projection())

    assert outcome.readback_status == "reconciled"
    assert len(sdk.save_calls) == 1


def test_bounded_scan_fails_closed_before_creating_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeWandb()
    sdk.listed_reports = [
        _ListedReport(
            id=f"other-{position}",
            display_name=f"Other {position}",
            description="unrelated",
            spec={},
            url=(
                "https://app.wandb.example/wandb/community-studies/reports/"
                f"Other--other-{position}"
            ),
        )
        for position in range(adapter._MAX_REPORT_SCAN + 1)
    ]

    with pytest.raises(
        adapter.WandbResearchReportPublicationError,
        match="bounded publication limit",
    ):
        _publisher(monkeypatch, sdk)(_projection())
    assert sdk.save_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("report_api", "wandb-workspaces.reports.v1"),
        ("report_api_version", "0.4.4"),
        ("api_stability", "stable"),
        ("report_width", "fluid"),
        ("projection_digest", "not-a-digest"),
        ("study_count", 0),
    ],
)
def test_incompatible_projection_fails_before_network(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    sdk = _FakeWandb()
    projection = replace(_projection(), **{field: value})

    with pytest.raises(adapter.WandbResearchReportPublicationError):
        _publisher(monkeypatch, sdk)(projection)
    assert sdk.api_calls == []


def test_markdown_escapes_untrusted_table_structure() -> None:
    projection = replace(
        _projection(),
        studies=(
            SimpleNamespace(
                **{
                    **vars(_study("study-a", rows=4, status="improved")),
                    "behavioral_recommendation": (
                        "first | second\nthird [unsafe](https://outside.example) <tag>"
                    ),
                }
            ),
            _study("study-b", rows=8, status="unchanged"),
        ),
    )

    rendered = adapter._report_markdown(
        projection,
        marker=adapter._projection_marker(projection.projection_digest),
    )

    assert "first \\| second third" in rendered
    assert "first | second\nthird" not in rendered
    assert "\\[unsafe\\]\\(https://outside.example\\)" in rendered
    assert "\\<tag\\>" in rendered
    assert "[unsafe](https://outside.example)" not in rendered


def test_report_publisher_shares_the_process_evidence_routing_lock() -> None:
    assert adapter._PUBLICATION_LOCK is weave_support.EVIDENCE_ROUTING_LOCK


def test_report_url_accepts_the_sdk_canonical_padding_free_id_path() -> None:
    value = (
        "https://app.wandb.example/wandb/community-studies/reports/"
        "Community--VmlldzoxMjM"
    )

    assert (
        adapter._validated_report_url(
            value,
            target=_target(),
            entity="wandb",
            project_id="community-studies",
            report_id="VmlldzoxMjM=",
        )
        == value
    )


@pytest.mark.parametrize("padding_free_url", [False, True])
def test_padded_report_id_survives_real_adapter_readback(
    monkeypatch: pytest.MonkeyPatch,
    padding_free_url: bool,
) -> None:
    sdk = _FakeWandb()
    sdk.report_id_override = "VmlldzoxNzY2NjU4Nw=="
    sdk.padding_free_url = padding_free_url

    outcome = _publisher(monkeypatch, sdk)(_projection())

    assert outcome.report_id == sdk.report_id_override
    expected_suffix = (
        sdk.report_id_override.rstrip("=")
        if padding_free_url
        else sdk.report_id_override
    )
    assert outcome.report_url.endswith(f"--{expected_suffix}")
