from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml

from fugue.bench.context import (
    ContextRuntime,
    PreparedContext,
    TrialContext,
    bind_context,
    load_context_system,
    preflight_context,
)
from fugue.bench.context_contracts import resolve_context_capabilities
from fugue.bench.library import experiment_from_yaml
from fugue.bench.sources import (
    SKILL_SOURCE_ROOT,
    canonical_git_url,
    load_skill_source,
    validate_relative_source_path,
)

CandidateKind = Literal["skill", "context_system"]

_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_EXACT_VERSION_RE = re.compile(
    r"(?:^|[@=])v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$"
)
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_REFERENCE_SUFFIXES = {".json", ".md", ".toml", ".txt", ".yaml", ".yml"}
_SOURCE_METADATA_KEYS = {
    "repository": ("fugue-source-repository", "source-repository"),
    "path": ("fugue-source-path", "source-path"),
    "commit": ("fugue-source-commit", "source-commit"),
    "license": ("fugue-source-license", "source-license"),
}


@dataclass(frozen=True)
class CurationPolicy:
    version: int
    maximum_inactive_days: int
    minimum_stars: dict[CandidateKind, int]
    allowed_licenses: frozenset[str]
    verified_owners: frozenset[str]
    skill_experiments: frozenset[str]
    context_experiment: str
    context_capabilities: frozenset[str]
    skill_capabilities: frozenset[str]
    discovery_skill_repositories: tuple[str, ...]
    discovery_integration_registries: tuple[str, ...]

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> CurationPolicy:
        if int(data.get("version", 0)) != 1:
            raise ValueError("curation policy version must be 1")
        gates = _mapping(data.get("gates"), name="gates")
        stars = _mapping(gates.get("minimum_stars"), name="gates.minimum_stars")
        lanes = _mapping(data.get("lanes"), name="lanes")
        skills = _mapping(lanes.get("skills"), name="lanes.skills")
        context = _mapping(lanes.get("context_systems"), name="lanes.context_systems")
        discovery = _mapping(data.get("discovery"), name="discovery")
        skill_discovery = _mapping(
            discovery.get("skills"), name="discovery.skills"
        )
        integration_discovery = _mapping(
            discovery.get("integrations"), name="discovery.integrations"
        )
        policy = cls(
            version=1,
            maximum_inactive_days=_positive_int(
                gates.get("maximum_inactive_days"),
                name="gates.maximum_inactive_days",
            ),
            minimum_stars={
                "skill": _nonnegative_int(
                    stars.get("skill"), name="gates.minimum_stars.skill"
                ),
                "context_system": _nonnegative_int(
                    stars.get("context_system"),
                    name="gates.minimum_stars.context_system",
                ),
            },
            allowed_licenses=frozenset(
                _strings(gates.get("allowed_licenses"), name="allowed_licenses")
            ),
            verified_owners=frozenset(
                item.casefold()
                for item in _strings(
                    gates.get("verified_owners"), name="verified_owners"
                )
            ),
            skill_experiments=frozenset(
                _strings(skills.get("experiments"), name="skills.experiments")
            ),
            context_experiment=_required_string(
                context.get("experiment"), name="context_systems.experiment"
            ),
            context_capabilities=frozenset(
                _strings(
                    context.get("allowed_capabilities"),
                    name="context_systems.allowed_capabilities",
                )
            ),
            skill_capabilities=frozenset(
                _strings(
                    skills.get("allowed_capabilities"),
                    name="skills.allowed_capabilities",
                )
            ),
            discovery_skill_repositories=tuple(
                _strings(
                    skill_discovery.get("repositories"),
                    name="discovery.skills.repositories",
                )
            ),
            discovery_integration_registries=tuple(
                _strings(
                    integration_discovery.get("registries"),
                    name="discovery.integrations.registries",
                )
            ),
        )
        if not policy.allowed_licenses:
            raise ValueError("gates.allowed_licenses must not be empty")
        if not policy.skill_experiments:
            raise ValueError("lanes.skills.experiments must not be empty")
        return policy

    @classmethod
    def load(cls, path: Path) -> CurationPolicy:
        data = yaml.safe_load(path.read_text()) or {}
        return cls.from_data(_mapping(data, name=str(path)))


@dataclass(frozen=True)
class CandidateRecord:
    kind: CandidateKind
    repository: str
    path: str | None
    commit: str
    stars: int
    last_push: datetime
    archived: bool
    license: str
    install_reference: str
    capabilities: tuple[str, ...]
    target_experiment: str
    has_executable_files: bool = False
    requires_new_dependencies: bool = False
    requires_custom_provider: bool = False
    requires_new_dataset: bool = False

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> CandidateRecord:
        allowed = {
            "kind",
            "repository",
            "path",
            "commit",
            "stars",
            "last_push",
            "archived",
            "license",
            "install_reference",
            "capabilities",
            "target_experiment",
            "has_executable_files",
            "requires_new_dependencies",
            "requires_custom_provider",
            "requires_new_dataset",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown candidate field(s): {', '.join(unknown)}")
        kind = str(data.get("kind") or "").strip()
        if kind not in {"skill", "context_system"}:
            raise ValueError("candidate kind must be 'skill' or 'context_system'")
        repository = _required_string(data.get("repository"), name="repository")
        if not _REPOSITORY_RE.fullmatch(repository):
            raise ValueError("repository must use the owner/name form")
        raw_path = data.get("path")
        candidate_path = (
            validate_relative_source_path(str(raw_path).strip())
            if raw_path
            else None
        )
        capabilities = tuple(
            sorted(set(_strings(data.get("capabilities"), name="capabilities")))
        )
        return cls(
            kind=kind,  # type: ignore[arg-type]
            repository=repository,
            path=candidate_path,
            commit=_required_string(data.get("commit"), name="commit").lower(),
            stars=_integer(data.get("stars"), name="stars", minimum=0),
            last_push=_datetime(data.get("last_push"), name="last_push"),
            archived=_boolean(data.get("archived"), name="archived"),
            license=_required_string(data.get("license"), name="license"),
            install_reference=_required_string(
                data.get("install_reference"), name="install_reference"
            ),
            capabilities=capabilities,
            target_experiment=_required_string(
                data.get("target_experiment"), name="target_experiment"
            ),
            has_executable_files=_boolean(
                data.get("has_executable_files", False),
                name="has_executable_files",
            ),
            requires_new_dependencies=_boolean(
                data.get("requires_new_dependencies", False),
                name="requires_new_dependencies",
            ),
            requires_custom_provider=_boolean(
                data.get("requires_custom_provider", False),
                name="requires_custom_provider",
            ),
            requires_new_dataset=_boolean(
                data.get("requires_new_dataset", False),
                name="requires_new_dataset",
            ),
        )

    @property
    def source_key(self) -> str:
        repository = self.repository.casefold()
        if self.kind == "skill":
            return f"skill:{repository}:{(self.path or '').casefold()}"
        return f"context_system:{repository}"

    @property
    def marker(self) -> str:
        source_path = self.path or "-"
        return (
            f"fugue-curator:candidate={self.kind}:"
            f"{self.repository.casefold()}:{source_path.casefold()}@{self.commit}"
        )


@dataclass(frozen=True)
class CandidateDecision:
    candidate_marker: str
    eligible: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    official_popularity_exception: bool
    evaluated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evaluated_at"] = self.evaluated_at.isoformat().replace("+00:00", "Z")
        return data


def evaluate_candidate(
    candidate: CandidateRecord,
    policy: CurationPolicy,
    *,
    repo_root: Path | None = None,
    prior_markers: Sequence[str] = (),
    evaluated_at: datetime | None = None,
) -> CandidateDecision:
    now = _utc(evaluated_at or datetime.now(UTC))
    reasons: list[str] = []
    warnings: list[str] = []
    owner = candidate.repository.partition("/")[0].casefold()
    official_exception = (
        owner in policy.verified_owners
        and candidate.stars < policy.minimum_stars[candidate.kind]
    )

    if candidate.archived:
        reasons.append("repository_archived")
    if candidate.last_push > now:
        reasons.append("last_push_in_future")
    elif now - candidate.last_push > timedelta(days=policy.maximum_inactive_days):
        reasons.append("repository_inactive")
    if candidate.license not in policy.allowed_licenses:
        reasons.append("license_not_allowed")
    if not _FULL_COMMIT_RE.fullmatch(candidate.commit):
        reasons.append("commit_not_immutable")
    if not _is_immutable_install_reference(candidate):
        reasons.append("install_reference_not_immutable")
    if (
        candidate.stars < policy.minimum_stars[candidate.kind]
        and not official_exception
    ):
        reasons.append("popularity_below_threshold")
    if official_exception:
        warnings.append("verified_owner_popularity_exception")

    reasons.extend(_candidate_contract_reasons(candidate, policy))

    if candidate.requires_new_dependencies:
        reasons.append("new_dependencies_required")
    if candidate.requires_custom_provider:
        reasons.append("custom_provider_required")
    if candidate.requires_new_dataset:
        reasons.append("new_dataset_required")

    if repo_root and candidate.source_key in existing_source_keys(repo_root):
        reasons.append("source_already_present")
    if any(candidate.marker in marker for marker in prior_markers):
        reasons.append("prior_curator_pr")

    unique_reasons = tuple(sorted(set(reasons)))
    return CandidateDecision(
        candidate_marker=candidate.marker,
        eligible=not unique_reasons,
        reasons=unique_reasons,
        warnings=tuple(sorted(set(warnings))),
        official_popularity_exception=official_exception,
        evaluated_at=now,
    )


def _candidate_contract_reasons(
    candidate: CandidateRecord, policy: CurationPolicy
) -> list[str]:
    if candidate.kind == "skill":
        reasons = []
        if not candidate.path:
            reasons.append("skill_path_required")
        if candidate.target_experiment not in policy.skill_experiments:
            reasons.append("target_experiment_not_allowed")
        if not candidate.capabilities:
            reasons.append("skill_capabilities_required")
        elif not set(candidate.capabilities) <= policy.skill_capabilities:
            reasons.append("skill_capabilities_not_allowed")
        if candidate.has_executable_files:
            reasons.append("executable_skill_bundle")
        return reasons
    reasons = []
    if candidate.path:
        reasons.append("context_path_not_allowed")
    if candidate.target_experiment != policy.context_experiment:
        reasons.append("target_experiment_not_allowed")
    if not candidate.capabilities:
        reasons.append("context_capabilities_required")
    elif not set(candidate.capabilities) <= policy.context_capabilities:
        reasons.append("context_capabilities_not_allowed")
    return reasons


def existing_source_keys(repo_root: Path) -> frozenset[str]:
    keys: set[str] = set()
    skills_root = repo_root / "configs" / "fugue" / "skills"
    for skill_file in sorted(skills_root.glob("*/SKILL.md")):
        metadata = _skill_metadata(skill_file)
        repository = _metadata_value(metadata, "repository")
        source_path = _metadata_value(metadata, "path")
        if not repository or not source_path:
            repository, source_path = _skill_source_url(metadata)
        if repository and source_path and _REPOSITORY_RE.fullmatch(repository):
            keys.add(
                f"skill:{repository.casefold()}:{source_path.strip('/').casefold()}"
            )

    context_root = repo_root / "configs" / "fugue" / "context-systems"
    for context_file in sorted(context_root.glob("*.yaml")):
        data = yaml.safe_load(context_file.read_text()) or {}
        if not isinstance(data, dict):
            continue
        repository = _github_repository(str(data.get("source_url") or ""))
        if repository:
            keys.add(f"context_system:{repository.casefold()}")
    return frozenset(keys)


def validate_skill_bundle(
    skill_dir: Path, *, require_provenance: bool = True
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        from skills_ref import validate
    except ImportError:
        return ("skills-ref==0.1.1 is required for skill validation",)

    errors.extend(str(error) for error in validate(skill_dir))
    for path in sorted(skill_dir.rglob("*")):
        relative = path.relative_to(skill_dir)
        if path.is_symlink():
            errors.append(f"{relative}: symbolic links are not allowed")
            continue
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(skill_dir.resolve()):
            errors.append(f"{relative}: path escapes the skill bundle")
            continue
        if path.stat().st_mode & 0o111:
            errors.append(f"{relative}: executable files are not allowed")
        if relative.parts[0] == "references":
            if path.suffix.casefold() not in _SAFE_REFERENCE_SUFFIXES:
                errors.append(f"{relative}: unsupported reference file type")
            continue
        if relative.name == "SKILL.md" or relative.name.startswith(
            ("LICENSE", "COPYING", "NOTICE")
        ):
            continue
        errors.append(f"{relative}: only instructions and references are allowed")

    if require_provenance:
        metadata = _skill_metadata(skill_dir / "SKILL.md")
        repository = _metadata_value(metadata, "repository")
        source_path = _metadata_value(metadata, "path")
        commit = _metadata_value(metadata, "commit")
        license_id = _metadata_value(metadata, "license")
        if not repository or not _REPOSITORY_RE.fullmatch(repository):
            errors.append("missing valid metadata.fugue-source-repository")
        if not source_path:
            errors.append("missing metadata.fugue-source-path")
        if not commit or not _FULL_COMMIT_RE.fullmatch(commit):
            errors.append("missing immutable metadata.fugue-source-commit")
        if not license_id:
            errors.append("missing metadata.fugue-source-license")
    return tuple(sorted(set(errors)))


def validate_skill_proposal(
    candidate: CandidateRecord,
    *,
    source_path: Path,
    experiment_path: Path,
    repo_root: Path,
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_root = (repo_root / SKILL_SOURCE_ROOT).resolve()
    if source_path.is_symlink() or not source_path.resolve().is_relative_to(expected_root):
        return ("skill source declaration must be a regular safe allowlisted path",)
    skill_id = source_path.stem
    try:
        source = load_skill_source(skill_id, repo_root).source
    except (OSError, ValueError) as exc:
        return (str(exc),)
    if canonical_git_url(source.url).casefold() != (
        f"https://github.com/{candidate.repository}".casefold()
    ):
        errors.append("skill source URL does not match candidate evidence")
    if source.ref.casefold() != candidate.commit.casefold():
        errors.append("skill source commit does not match candidate evidence")
    if source.path != candidate.path:
        errors.append("skill source path does not match candidate evidence")
    if errors:
        return tuple(sorted(set(errors)))
    experiment = experiment_from_yaml(
        experiment_path.read_text(), item_id=experiment_path.stem
    )
    if experiment.id == candidate.target_experiment:
        errors.append("proposal must add a dedicated experiment")
    base_path = (
        repo_root
        / "configs"
        / "fugue"
        / "experiments"
        / f"{candidate.target_experiment}.yaml"
    )
    base = experiment_from_yaml(base_path.read_text(), item_id=base_path.stem)
    preserved_fields = (
        "manifest",
        "model",
        "builder_model",
        "judge_model",
        "harnesses",
        "n_attempts",
        "n_concurrent",
        "n_tasks",
        "environment",
        "artifacts",
        "verifier",
        "retry",
        "agent_kwargs",
        "agent_env",
        "integrations",
        "workloads",
    )
    for field in preserved_fields:
        if getattr(experiment, field) != getattr(base, field):
            errors.append(f"proposal experiment must preserve base {field}")
    variants = [variant for variant in experiment.variants if variant.enabled]
    baseline = [variant for variant in variants if not variant.skills]
    treatments = [
        variant for variant in variants if variant.skills == [skill_id]
    ]
    if len(variants) != 2 or len(baseline) != 1 or len(treatments) != 1:
        errors.append(
            "proposal experiment requires exactly one baseline and one treatment"
        )
    elif _variant_without_skills(baseline[0]) != _variant_without_skills(treatments[0]):
        errors.append("treatment must preserve the complete baseline control")
    return tuple(sorted(set(errors)))


async def validate_context_proposal(
    candidate: CandidateRecord,
    context_path: Path,
    experiment_path: Path,
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    errors: list[str] = []
    if context_path.is_symlink():
        return ("context proposal path may not be a symbolic link",)
    spec = load_context_system(context_path)
    if spec.provider != "fugue.bench.context:CommandContextProvider":
        errors.append("context proposal must use CommandContextProvider")
    if spec.enabled_by_default:
        errors.append("context proposal must set enabled_by_default: false")
    normalized_source = _normalized_github_source(spec.source_url or "")
    if not normalized_source:
        errors.append("context proposal requires a GitHub source_url")
    elif normalized_source.casefold() != (
        f"https://github.com/{candidate.repository}".casefold()
    ):
        errors.append("context source_url does not match candidate evidence")
    if spec.license != candidate.license:
        errors.append("context license does not match candidate evidence")
    if spec.capabilities != frozenset(candidate.capabilities):
        errors.append("context capabilities do not match candidate evidence")
    if not _is_exact_pin(spec.version):
        errors.append("context proposal version must be immutable")
    proposal_references = [spec.version, *_nested_strings(spec.config)]
    if not any(
        candidate.install_reference == reference
        or candidate.commit in reference.casefold()
        for reference in proposal_references
    ):
        errors.append("context install reference does not match candidate evidence")
    mcp_servers = [
        item
        for item in (spec.config.get("binding") or {}).get("mcp_servers") or []
        if isinstance(item, dict)
    ]
    if not any(
        candidate.install_reference in [str(value) for value in server.get("args") or []]
        and str(server.get("command") or "") in {"uvx", "npx"}
        for server in mcp_servers
    ):
        errors.append("context executable must use the exact evaluated install pin")

    experiment = experiment_from_yaml(
        experiment_path.read_text(), item_id=experiment_path.stem
    )
    assigned_workloads = [
        workload
        for workload in experiment.workloads
        if spec.id in workload.systems
    ]
    matching_variants = [
        variant for variant in experiment.variants if variant.context.system_id == spec.id
    ]
    matching_workloads = [
        workload
        for workload in assigned_workloads
        if any(
            resolve_context_capabilities(
                spec,
                delivery=variant.context.delivery,
                runner=workload.runner,
                additional=workload.required_capabilities,
            ).applicable
            for variant in matching_variants
        )
    ]
    if experiment.id != candidate.target_experiment:
        errors.append("context experiment does not match candidate evidence")
    if experiment.id != "repo-memory-impact":
        errors.append("context proposal must extend repo-memory-impact")
    if not matching_workloads:
        errors.append("context proposal is not assigned to an applicable workload")
    if len(matching_workloads) != len(assigned_workloads):
        errors.append("context proposal is assigned to an incompatible workload")
    if not matching_variants:
        errors.append("context proposal requires an experiment variant")
    elif any(variant.context.delivery not in spec.deliveries for variant in matching_variants):
        errors.append("context proposal variant selects an unsupported delivery")
    if any(spec.id in preset.systems for preset in experiment.presets):
        errors.append("context proposal must remain outside default presets")

    if errors:
        return tuple(sorted(set(errors)))
    runtime = ContextRuntime(repo_root, repo_root / ".fugue" / "curation-preflight", {})
    await preflight_context(spec, runtime)
    prepared = PreparedContext(spec.id, "curation", repo_root, {}, {})
    await bind_context(
        spec,
        prepared,
        TrialContext("repo-memory-impact", "curation", "curation", "codex"),
        runtime,
        delivery=matching_variants[0].context.delivery,
    )
    return tuple(sorted(set(errors)))


def _is_immutable_install_reference(candidate: CandidateRecord) -> bool:
    reference = candidate.install_reference.strip()
    if candidate.kind == "skill":
        return candidate.commit in reference.casefold()
    return _is_exact_pin(reference)


def _is_exact_pin(reference: str) -> bool:
    lowered = reference.strip().casefold()
    if not lowered or any(
        token in lowered for token in ("latest", "@main", "@master", "@head", "*")
    ):
        return False
    if any(operator in reference for operator in ("^", "~", ">", "<")):
        return False
    if any(_FULL_COMMIT_RE.fullmatch(part) for part in re.split(r"[^0-9a-f]", lowered)):
        return True
    if re.search(r"[0-9a-f]{40}", lowered):
        return True
    return bool(_EXACT_VERSION_RE.search(reference.strip()))


def _skill_metadata(skill_file: Path) -> dict[str, Any]:
    if not skill_file.is_file():
        return {}
    lines = skill_file.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        closing = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration:
        return {}
    frontmatter = yaml.safe_load("\n".join(lines[1:closing])) or {}
    if not isinstance(frontmatter, dict):
        return {}
    metadata = frontmatter.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _skill_source_url(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    source = str(metadata.get("source_url") or metadata.get("source") or "").strip()
    parsed = urlparse(source)
    repository = _github_repository(source)
    if not repository:
        return None, None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
        return repository, "/".join(parts[4:])
    return repository, None


def _metadata_value(metadata: dict[str, Any], field: str) -> str | None:
    for key in _SOURCE_METADATA_KEYS[field]:
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return None


def _github_repository(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"github.com", "www.github.com"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    repository = f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return repository if _REPOSITORY_RE.fullmatch(repository) else None


def _normalized_github_source(source_url: str) -> str | None:
    parsed = urlparse(source_url.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"github.com", "www.github.com"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        return None
    repository = f"{parts[0]}/{parts[1].removesuffix('.git')}"
    if not _REPOSITORY_RE.fullmatch(repository):
        return None
    return f"https://github.com/{repository}"


def _variant_without_skills(value: Any) -> dict[str, Any]:
    data = value.to_dict()
    for key in ("id", "label", "skills"):
        data.pop(key, None)
    return data


def _nested_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _nested_strings(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _nested_strings(child)]
    return []


def _read_candidate(path: str) -> CandidateRecord:
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    data = json.loads(text)
    return CandidateRecord.from_data(_mapping(data, name="candidate"))


def _read_prior_markers(path: Path | None) -> list[str]:
    if path is None:
        return []
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [line for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError("prior markers file must contain a JSON list or one marker per line")
    return [str(item) for item in data]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal Fugue curation checks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate one candidate JSON record")
    evaluate.add_argument("--candidate", required=True, help="JSON path or '-' for stdin")
    evaluate.add_argument(
        "--policy", default="configs/fugue/curation.yaml", type=Path
    )
    evaluate.add_argument("--repo-root", default=Path.cwd(), type=Path)
    evaluate.add_argument("--as-of")
    evaluate.add_argument("--prior-marker", action="append", default=[])
    evaluate.add_argument("--prior-markers-file", type=Path)
    skill = subparsers.add_parser("validate-skill")
    skill.add_argument("skill_dir", type=Path)
    skill.add_argument("--allow-missing-provenance", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            candidate = _read_candidate(args.candidate)
            policy = CurationPolicy.load(args.policy)
            as_of = _datetime(args.as_of, name="as_of") if args.as_of else None
            prior_markers = [
                *args.prior_marker,
                *_read_prior_markers(args.prior_markers_file),
            ]
            decision = evaluate_candidate(
                candidate,
                policy,
                repo_root=args.repo_root,
                prior_markers=prior_markers,
                evaluated_at=as_of,
            )
            print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
            return 0
        errors = validate_skill_bundle(
            args.skill_dir,
            require_provenance=not args.allow_missing_provenance,
        )
        print(json.dumps({"eligible": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _strings(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    values = [str(item).strip() for item in value]
    if any(not item for item in values):
        raise ValueError(f"{name} must not contain empty values")
    return values


def _required_string(value: Any, *, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _datetime(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = _required_string(value, name=name).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return _utc(parsed)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("evaluation time must include a timezone")
    return value.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
