from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from fugue.bench.candidates import stable_digest
from fugue.bench.cli import _parser
from fugue.bench.comparison import ComparisonResultV3
from fugue.bench.reporting import (
    build_scientific_report_bundle,
    build_visual_data_manifest,
    campaign_publication_files,
    read_scientific_report_bundle,
    visual_data_manifest_from_dict,
)
from fugue.bench.scientific_reports import (
    ArticlePublicationReceiptV1,
    CampaignMembershipV1,
    CampaignStudyMembershipV1,
    ScientificReportError,
    VisualAssetManifestV1,
    VisualAssetV1,
    article_publication_receipt_from_dict,
    build_campaign_report_index,
    build_scientific_report,
    build_study_report_index,
    campaign_membership_from_dict,
    campaign_report_index_from_dict,
    publication_receipt_from_dict,
    render_report_markdown,
    scientific_report_from_dict,
    study_report_index_from_dict,
)
from fugue.model_plane import EvidenceDestinationV1
from fugue.research.contracts import AttributionV1
from fugue.research.experiment_views import ExperimentViewV3, experiment_view_from_dict
from fugue.research.report_publication import (
    WandbReportPublisher,
    attach_study_report_index,
    publish_report,
    record_study_report_index,
)
from fugue.research.store import StudyStore

FIXTURE = Path("tests/fixtures/experiment-view-v3-study-console-golden.json")


def _destination(project: str) -> EvidenceDestinationV1:
    entity, name = project.split("/", 1)
    return EvidenceDestinationV1(
        entity=entity,
        project=name,
        api_base_url="https://api.wandb.ai",
        trace_base_url="https://trace.wandb.ai",
        app_base_url="https://wandb.ai",
    )


@lru_cache(maxsize=1)
def _canonical_result() -> ComparisonResultV3:
    """Reuse the repository's complete V3 fixture builder, not a partial mock."""

    path = Path("tests/test_loop_engineering.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "fugue_test_loop_engineering_fixture",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module._result()
    assert type(result) is ComparisonResultV3
    return result


def _membership(*studies) -> CampaignMembershipV1:
    return CampaignMembershipV1(
        schema_version=1,
        campaign_id="community-skill-case-studies-v1",
        studies=tuple(
            sorted(
                (
                    CampaignStudyMembershipV1(
                        study_id=study.study_id,
                        result_project=study.result_project,
                    )
                    for study in studies
                ),
                key=lambda item: item.study_id,
            )
        ),
    )


def _publication_content() -> bytes:
    return json.dumps(
        {"kind": "scientific_report", "report_digest": "a" * 64},
        sort_keys=True,
    ).encode()


class SyntheticComparisonResult(ComparisonResultV3):
    def to_dict(self):
        return dict(self._synthetic_dict)


def _result(
    *, finding: str = "The candidate improved one locked outcome."
) -> ComparisonResultV3:
    result = object.__new__(SyntheticComparisonResult)
    pairs = (
        SimpleNamespace(
            task_id="task-a", attempt=1, status="improved", dimension_changes=()
        ),
        SimpleNamespace(
            task_id="task-b", attempt=1, status="improved", dimension_changes=()
        ),
    )
    values = {
        "comparison_id": "skill-upgrade-study",
        "improved": 2,
        "regressed": 0,
        "mixed": 0,
        "unchanged": 0,
        "incomplete": 0,
        "result_digest": "a" * 64,
        "qualification_digest": "b" * 64,
        "paired_cases": pairs,
        "behavioral_summary": SimpleNamespace(
            status="improved",
            supported_claim=finding,
            critical_blockers=(),
            limitations=("Two attempts per task.",),
            next_action="Advance to the sealed holdout.",
        ),
        "decision": SimpleNamespace(
            evidence_grade="A",
            limitations=("This is not a universal Skill ranking.",),
            next_action="Advance to the sealed holdout.",
        ),
        "task_validity": (
            SimpleNamespace(task_id="task-a", status="valid", blockers=()),
            SimpleNamespace(task_id="task-b", status="valid", blockers=()),
        ),
        "integrity": {"status": "reconciled"},
        "limitations": ("Exact revisions only.",),
        "judge_summary": {"status": "advisory"},
        "mechanism_summary": {"skill_opened": 4},
        "operational_summary": {"cost_status": "reconciled"},
        "evidence_topology": SimpleNamespace(
            source_destination=SimpleNamespace(project_slug="wandb/task-source"),
            result_destination=SimpleNamespace(
                project_slug="wandb/study-results",
                app_base_url="https://wandb.ai",
            ),
        ),
        "runtime_locks": (SimpleNamespace(digest="c" * 64),),
        "supersedes": (SimpleNamespace(result_digest="d" * 64),),
        "cohort_lineage": {
            "arms": {
                "baseline": {
                    "behavior_digest": "e" * 64,
                    "source_revisions": [
                        {
                            "kind": "skill",
                            "id": "public-skill",
                            "version_identity": "git:1111111",
                            "runtime_digest": "1" * 64,
                            "lock_digest": "2" * 64,
                        }
                    ],
                },
                "candidate": {
                    "behavior_digest": "f" * 64,
                    "source_revisions": [
                        {
                            "kind": "skill",
                            "id": "public-skill",
                            "version_identity": "git:2222222",
                            "runtime_digest": "3" * 64,
                            "lock_digest": "4" * 64,
                        }
                    ],
                },
            }
        },
    }
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(
        result,
        "_synthetic_dict",
        {
            "schema_version": 3,
            "comparison_id": result.comparison_id,
            "qualification_digest": result.qualification_digest,
            "result_digest": result.result_digest,
        },
    )
    return result


class FakePublisher:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def publish(self, *, request, content: bytes, markdown: str, artifact_files):
        self.calls.append(
            {
                "request": dict(request),
                "content": content,
                "markdown": markdown,
                "artifact_files": dict(artifact_files),
            }
        )
        destination = request["publication_destination"]
        project = f"{destination['entity']}/{destination['project']}"
        identity = dict(request)
        identity.pop("artifact_manifest")
        publication_id = stable_digest(identity)
        artifact_name = f"fugue-report-{publication_id[:20]}"
        artifact_ref = f"wandb-artifact://{project}/{artifact_name}:v0"
        return {
            "publication_run_id": "fugue-report-run",
            "artifact_url": f"https://wandb.ai/{project}/artifacts/{artifact_name}/v0",
            "artifact_ref": artifact_ref,
            "artifact_version": "v0",
            "artifact_digest": "artifact-digest-v0",
            "report_url": f"https://wandb.ai/{project}/reports/report",
            "publisher_id": "fake-wandb",
            "published_at": "2026-08-06T12:00:00+00:00",
        }


def _publish(tmp_path: Path):
    result = _canonical_result()
    source = tmp_path / "canonical-result.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle = build_scientific_report_bundle(source, tmp_path / "bundle")
    report = bundle.report
    artifact_files = bundle.artifact_files
    publisher = FakePublisher()
    first = publish_report(
        report,
        publisher=publisher,
        artifact_files=artifact_files,
        source_result=result,
        ledger_root=tmp_path,
    )
    second = publish_report(
        report,
        publisher=publisher,
        artifact_files=artifact_files,
        source_result=result,
        ledger_root=tmp_path,
    )
    return report, first, second, publisher


def test_v3_report_is_thin_sanitized_and_digest_bound() -> None:
    report = build_scientific_report(
        _result(finding="Improved; api_key=do-not-publish-this-value")
    )
    serialized = report.to_dict()

    assert scientific_report_from_dict(serialized) == report
    assert "do-not-publish" not in json.dumps(serialized)
    assert "paired_cases" not in serialized
    assert "mechanism_summary" not in serialized
    assert set(report.narrative) == {"what", "why", "how", "finding", "next_action"}
    assert "task is the inference unit" in report.narrative["how"]
    assert "aligned paired attempts" in report.narrative["what"]
    assert any("population sample" in item for item in report.limitations)
    assert any("universal Skill ranking" in item for item in report.limitations)
    assert report.named_blockers == ()
    assert report.exact_revisions == (
        "baseline:skill:public-skill@git:1111111",
        "candidate:skill:public-skill@git:2222222",
    )
    assert {item.status for item in report.claim_ledger} >= {
        "supported",
        "advisory",
        "descriptive",
    }

    tampered = dict(serialized)
    tampered["narrative"] = {**serialized["narrative"], "finding": "Different"}
    with pytest.raises(ScientificReportError, match="digest does not match"):
        scientific_report_from_dict(tampered)

    unresolved = _result()
    object.__setattr__(unresolved, "integrity", {"status": "invalid"})
    object.__setattr__(unresolved, "judge_summary", {"status": "not_used"})
    blocked = build_scientific_report(unresolved)
    by_id = {item.id: item.status for item in blocked.claim_ledger}
    assert by_id["deterministic-outcome"] == "blocked"
    assert by_id["judge-evidence"] == "unavailable"

    object.__setattr__(
        unresolved.behavioral_summary,
        "critical_blockers",
        ("task-b: package validity failed",),
    )
    blocked_with_name = build_scientific_report(unresolved)
    assert blocked_with_name.named_blockers == (
        "task-b: package validity failed",
    )
    assert "package validity" in render_report_markdown(blocked_with_name)


def test_publication_and_indexes_recompute_one_digest_chain(tmp_path: Path) -> None:
    report, first, second, publisher = _publish(tmp_path)

    assert first == second
    assert len(publisher.calls) == 1
    assert first.publication_id
    assert first.publication_bundle_digest
    assert first.artifact_manifest_digest
    assert first.artifact_ref.endswith(f":{first.artifact_version}")
    assert first.artifact_digest == "artifact-digest-v0"
    request = publisher.calls[0]["request"]
    assert request["run_kind"] == "report_only"
    assert request["excluded_from_task_inputs"] is True
    assert request["excluded_from_evaluation_counts"] is True
    assert (
        request["publication_destination"]["project_slug"]
        not in request["task_source_projects"]
    )
    assert request["publication_destination"]["api_base_url"] == (
        "https://api.wandb.ai"
    )

    study = build_study_report_index(
        report,
        first,
        result_url=(
            f"https://wandb.ai/{report.result_project}/artifacts/result"
        ),
        study_console_url=(
            "http://127.0.0.1:18080/?research_id=campaign&study_id="
            f"{report.comparison_id}"
        ),
        weave_url=f"https://wandb.ai/{report.result_project}/weave/evaluations/eval",
        visibility="public",
    )
    campaign = build_campaign_report_index(
        "community-skill-case-studies-v1",
        _destination("wandb/fugue-community-skill-case-studies-v1"),
        _membership(study),
        (study,),
    )

    assert study_report_index_from_dict(study.to_dict()) == study
    assert campaign_report_index_from_dict(campaign.to_dict()) == campaign
    assert study.summary.result_digest == report.source_result_digest
    assert study.reports[0] == first
    assert study.navigation_links
    assert next(
        item for item in study.reports[0].related_links if item.kind == "artifact"
    ).content_sha256 == first.artifact_manifest_digest
    assert campaign.studies[0].index_digest == study.index_digest
    assert campaign.no_pooled_ranking is True

    unsafe = first.to_dict()
    unsafe["publication"]["url"] = "https://wandb.ai/report?api_key=secret"
    with pytest.raises(ScientificReportError, match="credential parameters"):
        publication_receipt_from_dict(unsafe)


def test_publication_rejects_self_consistent_report_rewrite(tmp_path: Path) -> None:
    report, _first, _second, publisher = _publish(tmp_path / "accepted")
    tampered = replace(
        report,
        narrative={**report.narrative, "finding": "A rewritten conclusion."},
        report_digest="",
    )
    artifact_files = publisher.calls[0]["artifact_files"]

    with pytest.raises(
        ScientificReportError,
        match="does not recompute from ComparisonResultV3",
    ):
        publish_report(
            tampered,
            publisher=FakePublisher(),
            artifact_files=artifact_files,
            source_result=_canonical_result(),
            ledger_root=tmp_path / "tampered-ledger",
        )


def test_publication_rejects_remote_links_outside_bound_destination(
    tmp_path: Path,
) -> None:
    report, _first, _second, accepted_publisher = _publish(tmp_path / "accepted")

    class CrossProjectPublisher(FakePublisher):
        def publish(self, **kwargs):
            response = dict(super().publish(**kwargs))
            response["report_url"] = "https://wandb.ai/wandb/other/reports/report"
            return response

    with pytest.raises(ScientificReportError, match="outside the publication"):
        publish_report(
            report,
            publisher=CrossProjectPublisher(),
            artifact_files=accepted_publisher.calls[0]["artifact_files"],
            source_result=_canonical_result(),
            ledger_root=tmp_path / "cross-project-ledger",
        )


def test_study_index_keeps_one_exact_receipt_and_navigation_is_downstream(
    tmp_path: Path,
) -> None:
    report, receipt, _second, _publisher = _publish(tmp_path)
    study = build_study_report_index(
        report,
        receipt,
        result_url=f"https://wandb.ai/{report.result_project}/artifacts/result",
        study_console_url=f"http://localhost:18080/studies/{report.comparison_id}",
        weave_url=f"https://wandb.ai/{report.result_project}/weave",
    )
    raw = study.to_dict()
    raw["reports"].append(deepcopy(raw["reports"][0]))
    raw["report_count"] = 2
    raw["report_only_run_ids"] = [receipt.publication_run_id] * 2
    raw["index_digest"] = ""

    assert study.reports == (receipt,)
    assert all(link not in receipt.related_links for link in study.navigation_links)
    with pytest.raises(ScientificReportError, match="exactly one active"):
        study_report_index_from_dict(raw)


def test_campaign_membership_is_independent_complete_and_source_isolated(
    tmp_path: Path,
) -> None:
    locked = campaign_membership_from_dict(
        json.loads(
            Path(
                "examples/comparisons/community-skill-selected-v1/"
                "campaign-membership.lock.json"
            ).read_text(encoding="utf-8")
        )
    )
    assert len(locked.studies) == 3
    assert {item.study_id for item in locked.studies} == {
        "anthropic-skill-creator-measurement-pilot-v1",
        "superpowers-writing-plans-measurement-pilot-v2",
        "vercel-react-practices-measurement-pilot-v1",
    }
    with pytest.raises(ScientificReportError, match="result projects must be unique"):
        CampaignMembershipV1(
            schema_version=1,
            campaign_id="duplicate-projects",
            studies=(
                CampaignStudyMembershipV1(
                    study_id="lane-a",
                    result_project="wandb/shared-results",
                ),
                CampaignStudyMembershipV1(
                    study_id="lane-b",
                    result_project="wandb/shared-results",
                ),
            ),
        )

    report, receipt, _second, _publisher = _publish(tmp_path)
    study = build_study_report_index(
        report,
        receipt,
        result_url=f"https://wandb.ai/{report.result_project}/artifacts/result",
        study_console_url=f"http://localhost:18080/studies/{report.comparison_id}",
        weave_url=f"https://wandb.ai/{report.result_project}/weave",
    )
    membership = CampaignMembershipV1(
        schema_version=1,
        campaign_id="community-skill-case-studies-v1",
        studies=tuple(
            sorted(
                (
                    CampaignStudyMembershipV1(
                        study_id=study.study_id,
                        result_project=study.result_project,
                    ),
                    CampaignStudyMembershipV1(
                        study_id="lane-b",
                        result_project="wandb/lane-b-results",
                    ),
                    CampaignStudyMembershipV1(
                        study_id="lane-c",
                        result_project="wandb/lane-c-results",
                    ),
                ),
                key=lambda item: item.study_id,
            )
        ),
    )
    incomplete = build_campaign_report_index(
        "community-skill-case-studies-v1",
        _destination("wandb/fugue-community-skill-case-studies-v1"),
        membership,
        (study,),
    )
    assert incomplete.complete is False
    with pytest.raises(ScientificReportError, match="incomplete campaign"):
        publish_report(
            incomplete,
            publisher=FakePublisher(),
            artifact_files=campaign_publication_files(incomplete),
            ledger_root=tmp_path / "campaign-ledger",
        )
    with pytest.raises(ScientificReportError, match="task-source project"):
        build_campaign_report_index(
            "community-skill-case-studies-v1",
            _destination(report.source_project),
            _membership(study),
            (study,),
        )


class FakeArtifact:
    def __init__(self, *, name: str, type: str, metadata: dict) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[str] = []

    def new_file(self, name: str, *, mode: str):
        assert mode == "wb"
        self.files.append(name)
        return io.BytesIO()


class FakeRun:
    def __init__(self) -> None:
        self.id = "stable-report-run"
        self.summary: dict[str, object] = {}
        self.artifacts = 0
        self.finish_codes: list[int] = []

    def log_artifact(self, artifact):
        self.artifacts += 1
        return SimpleNamespace(
            url=(
                "https://wandb.ai/wandb/study-results/artifacts/"
                f"{artifact.name}/v0"
            ),
            qualified_name=f"wandb/study-results/{artifact.name}:v0",
            version="v0",
            digest="fake-artifact-digest-v0",
            wait=lambda: None,
        )

    def finish(self, *, exit_code: int) -> None:
        self.finish_codes.append(exit_code)


class FakeWandb:
    Artifact = FakeArtifact
    Settings = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_calls: list[dict] = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


class FakeRemoteReport:
    created = 0

    def __init__(self, **_kwargs) -> None:
        type(self).created += 1
        self.url = "https://wandb.ai/wandb/study-results/reports/report"

    def save(self) -> None:
        return None


def test_wandb_adapter_uses_one_report_only_identity() -> None:
    FakeRemoteReport.created = 0
    wandb = FakeWandb()
    reports = SimpleNamespace(
        Report=FakeRemoteReport,
        TableOfContents=lambda: object(),
        MarkdownBlock=lambda **kwargs: kwargs,
    )
    publisher = WandbReportPublisher(
        wandb_module=wandb,
        reports_module=reports,
        now=lambda: __import__("datetime").datetime(
            2026, 8, 6, tzinfo=__import__("datetime").UTC
        ),
    )
    content = _publication_content()
    request = _wandb_publication_request(content)

    first = publisher.publish(
        request=request,
        content=content,
        markdown="# Report",
        artifact_files={"report.json": content},
    )
    second = publisher.publish(
        request=request,
        content=content,
        markdown="# Report",
        artifact_files={"report.json": content},
    )

    assert first == second
    assert len({call["id"] for call in wandb.init_calls}) == 1
    assert all(call["job_type"] == "scientific-report" for call in wandb.init_calls)
    assert all(
        call["config"]["fugue"]["run_kind"] == "report_only"
        for call in wandb.init_calls
    )
    assert all(
        call["settings"].base_url == "https://api.wandb.ai"
        for call in wandb.init_calls
    )
    assert wandb.run.artifacts == 1
    assert FakeRemoteReport.created == 1


def test_wandb_adapter_rejects_destination_identity_drift() -> None:
    content = _publication_content()
    request = _wandb_publication_request(content)
    request["publication_destination"] = dict(request["publication_destination"])
    request["publication_destination"]["api_base_url"] = "https://api.other.test"

    with pytest.raises(ScientificReportError, match="destination is invalid"):
        WandbReportPublisher(
            wandb_module=FakeWandb(),
            reports_module=SimpleNamespace(),
        ).publish(
            request=request,
            content=content,
            markdown="# Report",
            artifact_files={"report.json": content},
        )


def test_wandb_adapter_rejects_run_routed_to_another_project() -> None:
    content = _publication_content()
    wandb = FakeWandb()
    wandb.run.entity = "wandb"
    wandb.run.project = "other-project"

    with pytest.raises(ScientificReportError, match="Run project disagrees"):
        WandbReportPublisher(
            wandb_module=wandb,
            reports_module=SimpleNamespace(),
        ).publish(
            request=_wandb_publication_request(content),
            content=content,
            markdown="# Report",
            artifact_files={"report.json": content},
        )


def _wandb_publication_request(content: bytes) -> dict:
    manifest = {
        "report.json": {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    }
    return {
        "schema_version": 2,
        "scope_kind": "study",
        "scope_id": "skill-upgrade-study",
        "document_kind": "scientific_report",
        "document_digest": "a" * 64,
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "publication_bundle_digest": "b" * 64,
        "artifact_manifest_digest": stable_digest(manifest),
        "source_artifact_digests": ["a" * 64],
        "publication_destination": _destination("wandb/study-results").to_dict(),
        "task_source_projects": ["wandb/task-source"],
        "run_kind": "report_only",
        "excluded_from_task_inputs": True,
        "excluded_from_evaluation_counts": True,
        "artifact_manifest": manifest,
    }


class RecoverablePublication:
    Settings = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))

    def __init__(self, *, crash_artifact: bool = False, crash_report: bool = False):
        self.run = FakeRun()
        self.run.id = "stable-report-run"
        self.run.log_artifact = self.log_artifact
        self.crash_artifact = crash_artifact
        self.crash_report = crash_report
        self.artifact_attempts = 0
        self.report_attempts = 0
        self.artifacts: dict[str, object] = {}
        self.reports: list[object] = []
        self.init_ids: list[str] = []

    def init(self, **kwargs):
        self.init_ids.append(kwargs["id"])
        return self.run

    def Artifact(self, *, name: str, type: str, metadata: dict):
        return FakeArtifact(name=name, type=type, metadata=metadata)

    def Api(self, *, overrides=None):
        assert overrides == {"base_url": "https://api.wandb.ai"}
        owner = self

        class Api:
            def artifact_exists(self, ref: str, *, type: str) -> bool:
                return f"{ref}:{type}" in owner.artifacts

            def artifact(self, ref: str, *, type: str):
                return owner.artifacts[f"{ref}:{type}"]

            def reports(self, _project: str, *, name: str):
                return [item for item in owner.reports if item.title == name]

        return Api()

    def log_artifact(self, artifact):
        self.artifact_attempts += 1
        ref = f"wandb/study-results/{artifact.name}:latest:{artifact.type}"
        remote = SimpleNamespace(
            url=(
                "https://wandb.ai/wandb/study-results/artifacts/"
                f"{artifact.name}/v0"
            ),
            metadata=artifact.metadata,
            qualified_name=f"wandb/study-results/{artifact.name}:v0",
            version="v0",
            digest="recoverable-artifact-digest-v0",
            wait=lambda: None,
        )
        self.artifacts[ref] = remote
        if self.crash_artifact:
            self.crash_artifact = False
            raise RuntimeError("crash after artifact side effect")
        return remote

    def report_module(self):
        owner = self

        class Report:
            def __init__(self, **kwargs) -> None:
                self.title = kwargs["title"]
                self.description = kwargs["description"]
                self.url = "https://wandb.ai/wandb/study-results/reports/report"

            def save(self) -> None:
                owner.report_attempts += 1
                owner.reports.append(self)
                if owner.crash_report:
                    owner.crash_report = False
                    raise RuntimeError("crash after report side effect")

        return SimpleNamespace(
            Report=Report,
            TableOfContents=lambda: object(),
            MarkdownBlock=lambda **kwargs: kwargs,
        )


@pytest.mark.parametrize("phase", ["run", "artifact", "report", "receipt"])
def test_wandb_publication_resumes_after_each_durable_phase(phase: str) -> None:
    remote = RecoverablePublication()
    content = _publication_content()
    seen = False

    def crash_once(current: str) -> None:
        nonlocal seen
        if current == phase and not seen:
            seen = True
            raise RuntimeError(f"crash after {phase}")

    publisher = WandbReportPublisher(
        wandb_module=remote,
        reports_module=remote.report_module(),
        phase_observer=crash_once,
    )
    with pytest.raises(RuntimeError, match=f"crash after {phase}"):
        publisher.publish(
            request=_wandb_publication_request(content),
            content=content,
            markdown="# Report",
            artifact_files={"report.json": content},
        )
    response = WandbReportPublisher(
        wandb_module=remote,
        reports_module=remote.report_module(),
    ).publish(
        request=_wandb_publication_request(content),
        content=content,
        markdown="# Report",
        artifact_files={"report.json": content},
    )

    assert response["publication_run_id"] == "stable-report-run"
    assert len(set(remote.init_ids)) == 1
    assert remote.artifact_attempts == 1
    assert remote.report_attempts == 1
    assert remote.run.summary["fugue.report_publication_v1"]["receipt"][
        "status"
    ] == "completed"


@pytest.mark.parametrize("effect", ["artifact", "report"])
def test_wandb_publication_recovers_a_completed_remote_side_effect(
    effect: str,
) -> None:
    remote = RecoverablePublication(
        crash_artifact=effect == "artifact",
        crash_report=effect == "report",
    )
    content = _publication_content()
    request = _wandb_publication_request(content)
    publisher = WandbReportPublisher(
        wandb_module=remote,
        reports_module=remote.report_module(),
    )
    with pytest.raises(RuntimeError, match=f"crash after {effect} side effect"):
        publisher.publish(
            request=request,
            content=content,
            markdown="# Report",
            artifact_files={"report.json": content},
        )
    response = publisher.publish(
        request=request,
        content=content,
        markdown="# Report",
        artifact_files={"report.json": content},
    )

    assert response["publication_run_id"] == "stable-report-run"
    assert remote.artifact_attempts == 1
    assert remote.report_attempts == 1
    assert len(remote.artifacts) == 1
    assert len(remote.reports) == 1


def test_article_receipt_is_append_only_and_does_not_mutate_campaign_facts(
    tmp_path: Path,
) -> None:
    report, publication, _second, _publisher = _publish(tmp_path)
    study = build_study_report_index(
        report,
        publication,
        result_url=f"https://wandb.ai/{report.result_project}/artifacts/result",
        study_console_url=f"http://localhost:18080/studies/{report.comparison_id}",
        weave_url=f"https://wandb.ai/{report.result_project}/weave/evaluations/eval",
        visibility="public",
    )
    campaign = build_campaign_report_index(
        "community-skill-case-studies-v1",
        _destination("wandb/fugue-community-skill-case-studies-v1"),
        _membership(study),
        (study,),
    )
    before = campaign.to_dict()
    receipt = ArticlePublicationReceiptV1(
        schema_version=1,
        kind="article_publication_receipt",
        campaign_index_digest=campaign.index_digest,
        article_url="https://wandb.ai/site/articles/do-agent-skills-get-better",
        article_source_sha256="9" * 64,
        published_at="2026-08-06T12:00:00+00:00",
        publisher_id="article-publisher-v1",
    )

    assert article_publication_receipt_from_dict(receipt.to_dict()) == receipt
    assert campaign.to_dict() == before
    assert receipt.campaign_index_digest == campaign.index_digest

    unsafe = receipt.to_dict()
    unsafe["article_url"] = "http://example.test/article"
    with pytest.raises(ScientificReportError, match="HTTPS or loopback HTTP"):
        article_publication_receipt_from_dict(unsafe)


def test_offline_bundle_includes_and_recomputes_canonical_result(
    tmp_path: Path,
) -> None:
    result = _canonical_result()
    source = tmp_path / "source-result.json"
    source.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bundle = build_scientific_report_bundle(source, tmp_path / "bundle")
    replay = read_scientific_report_bundle(tmp_path / "bundle")

    assert bundle.manifest == replay.manifest
    assert set(replay.artifact_files) >= {
        "bundle.json",
        "report.json",
        "report.md",
        "result.json",
    }
    assert json.loads(replay.artifact_files["result.json"])["result_digest"] == (
        result.result_digest
    )
    visual_data = build_visual_data_manifest(replay)
    states = {panel.id: panel.evidence_state for panel in visual_data.panels}
    assert states["next-stage"] == "planned"
    assert states["paired-dimensions"] == "observed"
    next_stage = next(panel for panel in visual_data.panels if panel.id == "next-stage")
    assert set(next_stage.data) == {"description"}
    tampered_visual_data = visual_data.to_dict()
    tampered_visual_data["panels"][0]["evidence_state"] = "planned"
    with pytest.raises(ScientificReportError):
        visual_data_manifest_from_dict(tampered_visual_data)

    (tmp_path / "bundle" / "report.md").write_text("drift", encoding="utf-8")
    with pytest.raises(ScientificReportError, match="file drifted"):
        read_scientific_report_bundle(tmp_path / "bundle")


def test_visual_manifest_is_claim_bound_and_requires_animation_companions() -> None:
    result = _result()
    asset = VisualAssetV1(
        id="matrix",
        role="experiment_matrix",
        group_id="study",
        path="assets/matrix.svg",
        media_type="image/svg+xml",
        sha256="1" * 64,
        size_bytes=10,
        alt_text="Locked baseline and candidate experiment matrix.",
        claim_ids=("deterministic-outcome",),
    )
    manifest = VisualAssetManifestV1(
        schema_version=1,
        kind="visual_asset_manifest",
        source_result_digest=result.result_digest,
        assets=(asset,),
    )

    report = build_scientific_report(result, visual_assets=manifest)
    assert scientific_report_from_dict(report.to_dict()) == report
    assert report.visual_assets is not None

    animation = VisualAssetV1(
        id="animation",
        role="provenance_animation",
        group_id="film",
        path="assets/film.mp4",
        media_type="video/mp4",
        sha256="2" * 64,
        size_bytes=10,
        alt_text="Attempts reconcile into a paired conclusion.",
        claim_ids=("evidence-integrity",),
    )
    with pytest.raises(ScientificReportError, match="requires reduced-motion"):
        VisualAssetManifestV1(
            schema_version=1,
            kind="visual_asset_manifest",
            source_result_digest=result.result_digest,
            assets=(animation,),
        )


def test_offline_bundle_copies_only_digest_bound_visual_files(
    tmp_path: Path,
) -> None:
    result = _canonical_result()
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    visual_root = tmp_path / "visual-publication"
    (visual_root / "assets").mkdir(parents=True)
    content = b"<svg>locked experiment matrix</svg>"
    (visual_root / "assets" / "matrix.svg").write_bytes(content)
    manifest = VisualAssetManifestV1(
        schema_version=1,
        kind="visual_asset_manifest",
        source_result_digest=result.result_digest,
        assets=(
            VisualAssetV1(
                id="matrix",
                role="experiment_matrix",
                group_id="study",
                path="assets/matrix.svg",
                media_type="image/svg+xml",
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                alt_text="Exact fixed and varied experiment conditions.",
                claim_ids=("deterministic-outcome",),
            ),
        ),
    )
    manifest_path = visual_root / "visual-assets.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    bundle = build_scientific_report_bundle(
        source,
        tmp_path / "visual-bundle",
        visual_manifest_path=manifest_path,
    )

    assert bundle.artifact_files["assets/matrix.svg"] == content
    assert bundle.report.visual_assets == manifest
    assert bundle.manifest.visual_manifest_digest == manifest.manifest_digest


def test_offline_bundle_rejects_secret_bearing_result_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    result._synthetic_dict["api_key"] = "do-not-publish-this-secret-value"
    monkeypatch.setattr(
        "fugue.bench.reporting.read_comparison_result",
        lambda _path: result,
    )
    source = tmp_path / "result.json"
    source.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    with pytest.raises(ScientificReportError, match="sensitive text"):
        build_scientific_report_bundle(source, tmp_path / "bundle")


def test_report_cli_keeps_build_offline_and_publication_explicit() -> None:
    parser = _parser()
    build = parser.parse_args(
        [
            "result",
            "study-a",
            "--report-action",
            "build",
            "--report-out",
            "report-bundle",
        ]
    )
    assert build.report_action == "build"
    assert build.yes is False

    publish = parser.parse_args(
        [
            "result",
            "--report-action",
            "publish",
            "--report-bundle",
            "report-bundle",
            "--result-url",
            "https://wandb.ai/wandb/results/artifacts/result",
            "--study-console-url",
            "http://127.0.0.1:18080/?study_id=study",
            "--weave-url",
            "https://wandb.ai/wandb/results/weave",
            "--receipt-out",
            "receipt.json",
            "--index-out",
            "index.json",
        ]
    )
    assert publish.report_action == "publish"
    assert publish.yes is False


def test_study_index_round_trips_through_view_and_research_log(tmp_path: Path) -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    view = experiment_view_from_dict(raw)
    assert isinstance(view, ExperimentViewV3)
    assert json.loads(json.dumps(view.to_dict())) == raw
    assert view.result_digest == raw["result_digest"]
    assert view.qualification_digest == raw["qualification_digest"]
    assert view.runtime_lock_digest == raw["runtime_lock_digest"]
    assert view.baseline_source_identity.role == "baseline"
    assert view.candidate_source_identity.role == "candidate"
    report, receipt, _second, _publisher = _publish(tmp_path / "ledger")
    index = build_study_report_index(
        report,
        receipt,
        result_url=f"https://wandb.ai/{report.result_project}/artifacts/result",
        study_console_url=f"https://study.example.test/studies/{report.comparison_id}",
        weave_url=f"https://wandb.ai/{report.result_project}/weave/evaluations/eval",
    )
    projected = attach_study_report_index(view, index)
    assert projected.report_index == index.to_dict()
    assert experiment_view_from_dict(projected.to_dict()) == projected

    store = StudyStore(tmp_path / "research")
    store.create_study(
        study_id="research-a",
        title="Research",
        campaign_id="campaign-a",
        question="Does the Skill revision help?",
        attribution=AttributionV1(actor_type="human", name="reviewer"),
        operation_id="create-research-a",
    )
    replayed, event = record_study_report_index(
        store=store,
        research_id="research-a",
        view=view,
        index=index,
    )

    assert replayed.report_index == index.to_dict()
    assert event.summary["experiment_view"]["report_index"] == index.to_dict()
    assert {item.kind for item in event.evidence} == {"report", "artifact"}
    assert next(item for item in event.evidence if item.kind == "report").digest == (
        index.reports[0].report_sha256
    )
    assert next(item for item in event.evidence if item.kind == "artifact").digest == (
        index.reports[0].artifact_manifest_digest
    )


def test_v3_wire_rejects_swapped_roles_and_runtime_lock_drift() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

    swapped = deepcopy(raw)
    swapped["baseline_source_identity"], swapped["candidate_source_identity"] = (
        swapped["candidate_source_identity"],
        swapped["baseline_source_identity"],
    )
    with pytest.raises(ValueError, match="baseline source identity role"):
        experiment_view_from_dict(swapped)

    drifted = deepcopy(raw)
    drifted["runtime_lock_digest"] = "f" * 64
    with pytest.raises(ValueError, match="runtime lock digest does not recompute"):
        experiment_view_from_dict(drifted)


@pytest.mark.parametrize(
    "unsafe_url",
    (
        "https://wandb.ai/report#fragment",
        "http://wandb.ai/report",
        "https://wandb.ai/report?access_token=secret",
    ),
)
def test_report_links_reject_unsafe_navigation(unsafe_url: str, tmp_path: Path) -> None:
    _report, receipt, _second, _publisher = _publish(tmp_path)
    raw = receipt.to_dict()
    raw["publication"]["url"] = unsafe_url
    with pytest.raises(ScientificReportError):
        publication_receipt_from_dict(raw)
