from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from fugue.bench.candidates import stable_digest
from fugue.bench.library import validate_id
from fugue.research.agent_contracts import (
    CandidateRefV1,
    candidate_ref_from_dict,
    ensure_registered_candidate,
)
from fugue.research.contracts import ExperimentDraftV1, ResearchError

_SOURCE_FIELDS = {
    "allowed_experiments",
    "allowed_variants",
    "content_digest",
    "id",
    "kind",
    "path",
    "url",
}


@dataclass(frozen=True)
class RegisteredCandidateSource:
    id: str
    kind: str
    location: str
    content_digest: str | None
    allowed_experiments: tuple[str, ...]
    allowed_variants: tuple[str, ...]
    source_digest: str

    def safe_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "allowed_experiments": list(self.allowed_experiments),
            "allowed_variants": list(self.allowed_variants),
            "source_digest": self.source_digest,
        }


class CandidateSourceRegistry:
    """Operator-managed repositories and artifacts accepted from Agent drafts."""

    def __init__(self, sources: Iterable[RegisteredCandidateSource] = ()) -> None:
        values = tuple(sources)
        self._sources = {source.id: source for source in values}
        if len(self._sources) != len(values):
            raise ValueError("candidate source ids must be unique")

    @classmethod
    def from_file(cls, path: Path | None) -> CandidateSourceRegistry:
        if path is None:
            return cls()
        root = path.resolve().parent
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_mapping(raw, root=root)

    @classmethod
    def from_mapping(cls, raw: Any, *, root: Path) -> CandidateSourceRegistry:
        if not isinstance(raw, dict) or set(raw) != {"version", "sources"}:
            raise ValueError(
                "candidate source config requires only version and sources"
            )
        if raw["version"] != 1 or not isinstance(raw["sources"], list):
            raise ValueError("candidate source config version must be 1")
        return cls(_source(value, root) for value in raw["sources"])

    def catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            source.safe_dict()
            for source in sorted(self._sources.values(), key=lambda item: item.id)
        )

    def validate_draft(self, draft: ExperimentDraftV1) -> None:
        references = tuple(
            candidate_ref_from_dict(item) for item in draft.candidate_refs
        )
        by_source = {reference.repository_id: reference for reference in references}
        if len(by_source) != len(references):
            raise ResearchError(
                "duplicate_candidate_source",
                "experiment candidate source references must be unique",
            )
        selected_variants = draft.variants
        for reference in references:
            source = self._get(reference.repository_id)
            self._validate_reference(source, reference, draft, selected_variants)
        used_sources = self._validate_task_sources(draft.task_suite_draft, by_source)
        unused = sorted(set(by_source) - used_sources)
        if unused:
            raise ResearchError(
                "unused_candidate_source",
                "candidate references must affect a task resource: "
                + ", ".join(unused),
                category="policy",
            )

    def _get(self, source_id: str) -> RegisteredCandidateSource:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise ResearchError(
                "candidate_source_not_registered",
                f"candidate source is not registered: {source_id}",
                category="policy",
            ) from exc

    @staticmethod
    def _validate_reference(
        source: RegisteredCandidateSource,
        reference: CandidateRefV1,
        draft: ExperimentDraftV1,
        variants: Sequence[str],
    ) -> None:
        expected_kind = "git_commit" if source.kind == "git" else "artifact"
        if reference.source_kind != expected_kind:
            raise ResearchError(
                "candidate_source_kind_mismatch",
                "candidate reference kind does not match its registered source",
                category="policy",
            )
        if reference.source_digest != source.source_digest:
            raise ResearchError(
                "candidate_source_drift",
                "registered candidate source changed after the reference was created",
                category="policy",
            )
        if draft.experiment_id not in source.allowed_experiments:
            raise ResearchError(
                "candidate_experiment_mismatch",
                "candidate source is not registered for this experiment",
                category="policy",
            )
        ensure_registered_candidate(
            reference,
            experiment_id=draft.experiment_id,
            variants=variants or source.allowed_variants,
        )
        if (
            reference.registered_variant_id not in source.allowed_variants
            or source.content_digest is not None
            and reference.content_digest != source.content_digest
        ):
            raise ResearchError(
                "candidate_source_mismatch",
                "candidate reference does not match its registered source",
                category="policy",
            )
        if source.kind == "artifact" and reference.revision != source.content_digest:
            raise ResearchError(
                "candidate_revision_mismatch",
                "artifact candidate revision must equal its registered content digest",
                category="policy",
            )

    def _validate_task_sources(
        self,
        task_suite: Mapping[str, Any] | None,
        references: Mapping[str, CandidateRefV1],
    ) -> set[str]:
        if task_suite is None:
            return set()
        repositories = []
        resource_ids: set[str] = set()
        for task in task_suite.get("tasks") or ():
            if not isinstance(task, Mapping):
                continue
            environment = task.get("environment")
            if not isinstance(environment, Mapping):
                continue
            repository = environment.get("repository")
            if isinstance(repository, Mapping):
                repositories.append(repository)
            for part in task.get("prompt") or ():
                if isinstance(part, Mapping) and part.get("resource_profile_id"):
                    resource_ids.add(str(part["resource_profile_id"]))
        used: set[str] = set()
        for repository in repositories:
            url = str(repository.get("url") or "")
            commit = str(repository.get("commit") or "")
            source = next(
                (
                    item
                    for item in self._sources.values()
                    if item.kind == "git" and item.location == url
                ),
                None,
            )
            if source is None:
                raise ResearchError(
                    "candidate_source_not_registered",
                    "authored task repository is not in the operator source catalog",
                    category="policy",
                )
            reference = references.get(source.id)
            if reference is None or reference.revision != commit:
                raise ResearchError(
                    "candidate_revision_mismatch",
                    "authored task repository must match an immutable candidate reference",
                    category="policy",
                )
            used.add(source.id)
        for source_id in resource_ids:
            source = self._sources.get(source_id)
            if source is not None and source.kind == "artifact":
                if source_id not in references:
                    raise ResearchError(
                        "candidate_source_reference_required",
                        "registered candidate artifact requires an immutable reference",
                        category="policy",
                    )
                used.add(source_id)
        return used


def _source(raw: Any, root: Path) -> RegisteredCandidateSource:
    if not isinstance(raw, dict):
        raise ValueError("candidate source entry must be an object")
    unknown = sorted(set(raw) - _SOURCE_FIELDS)
    if unknown:
        raise ValueError("unknown candidate source fields: " + ", ".join(unknown))
    source_id = validate_id(str(raw.get("id") or ""), kind="candidate source id")
    kind = str(raw.get("kind") or "")
    if kind not in {"git", "artifact"}:
        raise ValueError("candidate source kind must be git or artifact")
    url = raw.get("url")
    path = raw.get("path")
    if kind == "git":
        if not isinstance(url, str) or path is not None:
            raise ValueError("git candidate sources require only a URL")
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("git candidate source URL must be public HTTPS")
        location = url
    else:
        if not isinstance(path, str) or url is not None:
            raise ValueError("artifact candidate sources require only a path")
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "artifact candidate source path escapes its config root"
            ) from exc
        if not resolved.is_file():
            raise ValueError("artifact candidate source must be a regular file")
        location = str(resolved)
    experiments = _ids(raw.get("allowed_experiments"), "allowed experiment")
    variants = _ids(raw.get("allowed_variants"), "allowed variant")
    content_digest = raw.get("content_digest")
    if content_digest is not None and not _digest(str(content_digest)):
        raise ValueError("candidate source content_digest must be a sha256 digest")
    if kind == "artifact":
        if content_digest is None:
            raise ValueError("artifact candidate source requires content_digest")
        if hashlib.sha256(Path(location).read_bytes()).hexdigest() != content_digest:
            raise ValueError("artifact candidate source content_digest does not match")
    identity = {
        "id": source_id,
        "kind": kind,
        "location": location,
        "content_digest": content_digest,
        "allowed_experiments": experiments,
        "allowed_variants": variants,
    }
    return RegisteredCandidateSource(
        id=source_id,
        kind=kind,
        location=location,
        content_digest=str(content_digest) if content_digest else None,
        allowed_experiments=experiments,
        allowed_variants=variants,
        source_digest=stable_digest(identity),
    )


def _ids(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}s must be a non-empty list")
    result = tuple(validate_id(str(item), kind=label) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label}s contain duplicates")
    return result


def _digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
