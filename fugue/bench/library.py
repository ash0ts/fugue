from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

CONFIG_ROOT = Path("configs") / "fugue"
PROMPTS_DIR = "prompts"
SKILLS_DIR = "skills"
EXPERIMENTS_DIR = "experiments"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class LibraryItem:
    id: str
    title: str
    path: str
    sha256: str


@dataclass(frozen=True)
class Prompt:
    id: str
    title: str
    body: str
    path: str
    sha256: str


@dataclass(frozen=True)
class Skill:
    id: str
    title: str
    body: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ContextSelection:
    system_id: str = "none"
    transport: Literal["portable", "native_mcp"] = "portable"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureVariant:
    id: str
    label: str
    prompt_id: str | None = None
    skill_ids: list[str] = field(default_factory=list)
    context: ContextSelection = field(default_factory=ContextSelection)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_env: dict[str, str] = field(default_factory=dict)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    verifier: dict[str, Any] = field(default_factory=dict)
    retry: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Any] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadSpec:
    id: str
    runner: str
    manifest: Path | None = None
    dataset: str | None = None
    systems: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    n_tasks: int | None = None
    n_attempts: int | None = None
    artifacts: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class PresetSpec:
    id: str
    workloads: list[str] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)
    harnesses: list[str] = field(default_factory=list)
    n_tasks: int | None = None
    n_attempts: int | None = None
    n_concurrent: int | None = None
    workload_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    title: str
    description: str = ""
    manifest: Path = Path("datasets/pilot.yaml")
    model: str | None = None
    builder_model: str | None = None
    judge_model: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    harnesses: list[str] = field(default_factory=list)
    variants: list[FeatureVariant] = field(default_factory=list)
    n_attempts: int | None = None
    n_concurrent: int | None = None
    n_tasks: int | None = None
    jobs_dir: Path | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Any] = field(default_factory=list)
    verifier: dict[str, Any] = field(default_factory=dict)
    retry: dict[str, Any] = field(default_factory=dict)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_env: dict[str, str] = field(default_factory=dict)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    workloads: list[WorkloadSpec] = field(default_factory=list)
    presets: list[PresetSpec] = field(default_factory=list)
    default_preset: str | None = None
    trace_content: str = "full"
    debug: bool = False
    quiet: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _paths_to_strings(asdict(self))


def library_root(repo_root: Path | None = None) -> Path:
    root = repo_root or Path.cwd()
    return root / CONFIG_ROOT


def validate_id(value: str, *, kind: str = "id") -> str:
    item_id = str(value or "").strip()
    if not _ID_RE.match(item_id):
        raise ValueError(
            f"invalid {kind} {value!r}; use letters, numbers, '.', '_' or '-'"
        )
    return item_id


def list_prompts(repo_root: Path | None = None) -> list[LibraryItem]:
    return _list_markdown_items(library_root(repo_root) / PROMPTS_DIR)


def get_prompt(item_id: str, repo_root: Path | None = None) -> Prompt:
    item_id = validate_id(item_id, kind="prompt id")
    path = library_root(repo_root) / PROMPTS_DIR / f"{item_id}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {item_id}")
    body = path.read_text()
    return Prompt(
        id=item_id,
        title=_title_from_markdown(body) or item_id,
        body=body,
        path=path.as_posix(),
        sha256=_file_sha256(path),
    )


def save_prompt(item_id: str, body: str, repo_root: Path | None = None) -> Prompt:
    item_id = validate_id(item_id, kind="prompt id")
    path = library_root(repo_root) / PROMPTS_DIR / f"{item_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_normalize_text(body))
    return get_prompt(item_id, repo_root)


def list_skills(repo_root: Path | None = None) -> list[LibraryItem]:
    root = library_root(repo_root) / SKILLS_DIR
    if not root.exists():
        return []
    items: list[LibraryItem] = []
    for skill_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        skill_path = skill_dir / "SKILL.md"
        if skill_path.is_file():
            items.append(
                _library_item(
                    skill_dir.name,
                    skill_path,
                    _title_from_markdown(skill_path.read_text()),
                )
            )
    return items


def get_skill(item_id: str, repo_root: Path | None = None) -> Skill:
    item_id = validate_id(item_id, kind="skill id")
    path = library_root(repo_root) / SKILLS_DIR / item_id / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"skill not found: {item_id}")
    body = path.read_text()
    return Skill(
        id=item_id,
        title=_title_from_markdown(body) or item_id,
        body=body,
        path=path.as_posix(),
        sha256=_file_sha256(path),
    )


def save_skill(item_id: str, body: str, repo_root: Path | None = None) -> Skill:
    item_id = validate_id(item_id, kind="skill id")
    path = library_root(repo_root) / SKILLS_DIR / item_id / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_normalize_text(body))
    return get_skill(item_id, repo_root)


def list_experiments(repo_root: Path | None = None) -> list[LibraryItem]:
    return _list_yaml_items(library_root(repo_root) / EXPERIMENTS_DIR)


def get_experiment(item_id: str, repo_root: Path | None = None) -> ExperimentSpec:
    item_id = validate_id(item_id, kind="experiment id")
    path = library_root(repo_root) / EXPERIMENTS_DIR / f"{item_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"experiment not found: {item_id}")
    experiment = experiment_from_yaml(path.read_text(), item_id=item_id)
    if experiment.id != item_id:
        raise ValueError(
            f"experiment file {path.name!r} declares mismatched id {experiment.id!r}"
        )
    return experiment


def get_experiment_text(item_id: str, repo_root: Path | None = None) -> str:
    item_id = validate_id(item_id, kind="experiment id")
    path = library_root(repo_root) / EXPERIMENTS_DIR / f"{item_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"experiment not found: {item_id}")
    return path.read_text()


def save_experiment(
    item_id: str, body: str, repo_root: Path | None = None
) -> ExperimentSpec:
    item_id = validate_id(item_id, kind="experiment id")
    experiment = experiment_from_yaml(body, item_id=item_id)
    if experiment.id != item_id:
        raise ValueError(
            f"experiment body id {experiment.id!r} does not match {item_id!r}"
        )
    path = library_root(repo_root) / EXPERIMENTS_DIR / f"{item_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_normalize_text(body))
    return get_experiment(item_id, repo_root)


def save_experiment_data(
    item_id: str, data: dict[str, Any], repo_root: Path | None = None
) -> ExperimentSpec:
    data = dict(data)
    data["id"] = item_id
    experiment = experiment_from_data(data, item_id=item_id)
    return save_experiment(item_id, experiment_to_yaml(experiment), repo_root)


def experiment_from_yaml(
    text: str, *, item_id: str | None = None
) -> ExperimentSpec:
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError("experiment YAML must be a mapping")
    return experiment_from_data(raw, item_id=item_id)


def experiment_from_data(
    raw: dict[str, Any], *, item_id: str | None = None
) -> ExperimentSpec:
    _reject_unknown(raw, ExperimentSpec, kind="experiment")
    experiment_id = validate_id(
        raw.get("id") or item_id or "experiment", kind="experiment id"
    )
    variants = _variants(raw)
    if not variants:
        variants = [FeatureVariant(id="baseline", label="Baseline")]
    workloads = _workloads(raw.get("workloads"))
    presets = _presets(raw.get("presets"))
    _require_unique([variant.id for variant in variants], kind="variant")
    _require_unique([workload.id for workload in workloads], kind="workload")
    _require_unique([preset.id for preset in presets], kind="preset")
    workload_ids = {workload.id for workload in workloads}
    for preset in presets:
        unknown = sorted(
            (set(preset.workloads) | set(preset.workload_overrides)) - workload_ids
        )
        if unknown:
            raise ValueError(
                f"preset {preset.id} overrides unknown workload(s): {', '.join(unknown)}"
            )
    default_preset = _optional_str(raw.get("default_preset"))
    if default_preset and default_preset not in {preset.id for preset in presets}:
        raise ValueError(f"unknown default preset: {default_preset}")
    return ExperimentSpec(
        id=experiment_id,
        title=str(raw.get("title") or experiment_id),
        description=str(raw.get("description") or ""),
        manifest=Path(str(raw.get("manifest") or "datasets/pilot.yaml")),
        model=_optional_str(raw.get("model")),
        builder_model=_optional_str(raw.get("builder_model")),
        judge_model=_optional_str(raw.get("judge_model")),
        run_name=_optional_str(raw.get("run_name")),
        tags=_string_list(raw.get("tags")),
        harnesses=_string_list(raw.get("harnesses")),
        variants=variants,
        n_attempts=_positive_int(raw.get("n_attempts"), kind="experiment n_attempts"),
        n_concurrent=_positive_int(
            raw.get("n_concurrent"), kind="experiment n_concurrent"
        ),
        n_tasks=_positive_int(raw.get("n_tasks"), kind="experiment n_tasks"),
        jobs_dir=Path(str(raw["jobs_dir"])) if raw.get("jobs_dir") else None,
        environment=_dict(raw.get("environment")),
        artifacts=_list(raw.get("artifacts")),
        verifier=_dict(raw.get("verifier")),
        retry=_dict(raw.get("retry")),
        agent_kwargs=_dict(raw.get("agent_kwargs")),
        agent_env={str(k): str(v) for k, v in _dict(raw.get("agent_env")).items()},
        mcp_servers=[_dict(item) for item in _list(raw.get("mcp_servers"))],
        workloads=workloads,
        presets=presets,
        default_preset=default_preset,
        trace_content=_trace_content(raw.get("trace_content")),
        debug=bool(raw.get("debug", False)),
        quiet=bool(raw.get("quiet", False)),
    )


def experiment_to_yaml(experiment: ExperimentSpec) -> str:
    return yaml.safe_dump(experiment.to_dict(), sort_keys=False)


def experiment_with_overrides(
    experiment: ExperimentSpec, **overrides: Any
) -> ExperimentSpec:
    data = experiment.to_dict()
    for key, value in overrides.items():
        if value in (None, "", []):
            continue
        data[key] = value
    return experiment_from_yaml(yaml.safe_dump(data), item_id=experiment.id)


def content_hashes_for_ids(
    *,
    prompt_ids: list[str],
    skill_ids: list[str],
    repo_root: Path | None = None,
) -> dict[str, dict[str, str]]:
    return {
        "prompts": {item_id: get_prompt(item_id, repo_root).sha256 for item_id in prompt_ids},
        "skills": {item_id: get_skill(item_id, repo_root).sha256 for item_id in skill_ids},
    }


def _variants(raw: dict[str, Any]) -> list[FeatureVariant]:
    values = raw.get("variants")
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("variants must be a list")
    return [_feature_variant(item, index) for index, item in enumerate(values, start=1)]


def _feature_variant(raw: Any, index: int) -> FeatureVariant:
    if not isinstance(raw, dict):
        raise ValueError("variant must be a mapping")
    _reject_unknown(raw, FeatureVariant, kind="variant")
    variant_id = validate_id(raw.get("id") or f"variant-{index}", kind="variant id")
    prompt_id = _optional_str(raw.get("prompt_id"))
    if prompt_id:
        validate_id(prompt_id, kind="prompt id")
    skill_ids = _string_list(raw.get("skill_ids"))
    for skill_id in skill_ids:
        validate_id(skill_id, kind="skill id")
    _require_unique(skill_ids, kind=f"variant {variant_id} skill")
    return FeatureVariant(
        id=variant_id,
        label=str(raw.get("label") or variant_id),
        prompt_id=prompt_id,
        skill_ids=skill_ids,
        context=_context_selection(raw.get("context")),
        agent_kwargs=_dict(raw.get("agent_kwargs")),
        agent_env={str(k): str(v) for k, v in _dict(raw.get("agent_env")).items()},
        mcp_servers=[_dict(item) for item in _list(raw.get("mcp_servers"))],
        environment=_dict(raw.get("environment")),
        verifier=_dict(raw.get("verifier")),
        retry=_dict(raw.get("retry")),
        artifacts=_list(raw.get("artifacts")),
        enabled=bool(raw.get("enabled", True)),
    )


def _context_selection(raw: Any) -> ContextSelection:
    if raw in (None, ""):
        return ContextSelection()
    if isinstance(raw, str):
        system_id = validate_id(raw, kind="context system id")
        return ContextSelection(system_id=system_id)
    if not isinstance(raw, dict):
        raise ValueError("variant context must be a string or mapping")
    _reject_unknown(raw, ContextSelection, kind="variant context")
    system_id = validate_id(raw.get("system_id") or "none", kind="context system id")
    transport = str(raw.get("transport") or "portable")
    if transport not in {"portable", "native_mcp"}:
        raise ValueError(f"unknown context transport: {transport}")
    return ContextSelection(
        system_id=system_id,
        transport=transport,
        config=_dict(raw.get("config")),
    )


def _workloads(raw: Any) -> list[WorkloadSpec]:
    values = _list(raw)
    workloads: list[WorkloadSpec] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError("workload must be a mapping")
        _reject_unknown(value, WorkloadSpec, kind="workload")
        workload_id = validate_id(
            value.get("id") or f"workload-{index}", kind="workload id"
        )
        runner = str(value.get("runner") or "harbor")
        if runner not in {"harbor", "retrieval", "sequence"}:
            raise ValueError(f"unknown workload runner: {runner}")
        workloads.append(
            WorkloadSpec(
                id=workload_id,
                runner=runner,
                manifest=Path(str(value["manifest"])) if value.get("manifest") else None,
                dataset=_optional_str(value.get("dataset")),
                systems=_string_list(value.get("systems")),
                required_capabilities=_string_list(value.get("required_capabilities")),
                n_tasks=_positive_int(
                    value.get("n_tasks"), kind=f"workload {workload_id} n_tasks"
                ),
                n_attempts=_positive_int(
                    value.get("n_attempts"),
                    kind=f"workload {workload_id} n_attempts",
                ),
                artifacts=_list(value.get("artifacts")),
            )
        )
    return workloads


def _presets(raw: Any) -> list[PresetSpec]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        values = [{"id": key, **_dict(value)} for key, value in raw.items()]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError("presets must be a mapping or list")
    presets: list[PresetSpec] = []
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ValueError("preset must be a mapping")
        _reject_unknown(value, PresetSpec, kind="preset")
        preset_id = validate_id(value.get("id") or f"preset-{index}", kind="preset id")
        presets.append(
            PresetSpec(
                id=preset_id,
                workloads=_string_list(value.get("workloads")),
                systems=_string_list(value.get("systems")),
                harnesses=_string_list(value.get("harnesses")),
                n_tasks=_positive_int(
                    value.get("n_tasks"), kind=f"preset {preset_id} n_tasks"
                ),
                n_attempts=_positive_int(
                    value.get("n_attempts"), kind=f"preset {preset_id} n_attempts"
                ),
                n_concurrent=_positive_int(
                    value.get("n_concurrent"),
                    kind=f"preset {preset_id} n_concurrent",
                ),
                workload_overrides=_workload_overrides(
                    value.get("workload_overrides"), preset_id
                ),
            )
        )
    return presets


def _library_item(item_id: str, path: Path, title: str | None) -> LibraryItem:
    return LibraryItem(
        id=item_id,
        title=title or item_id,
        path=path.as_posix(),
        sha256=_file_sha256(path),
    )


def _list_markdown_items(root: Path) -> list[LibraryItem]:
    if not root.exists():
        return []
    return [
        _library_item(path.stem, path, _title_from_markdown(path.read_text()))
        for path in sorted(root.glob("*.md"))
        if path.is_file()
    ]


def _list_yaml_items(root: Path) -> list[LibraryItem]:
    if not root.exists():
        return []
    return [
        _library_item(path.stem, path, _experiment_title(path))
        for path in sorted(root.glob("*.yaml"))
        if path.is_file()
    ]


def _experiment_title(path: Path) -> str:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return path.stem
    return (
        str(raw.get("title") or raw.get("id") or path.stem)
        if isinstance(raw, dict)
        else path.stem
    )


def _title_from_markdown(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_text(body: str) -> str:
    text = str(body or "")
    return text if text.endswith("\n") else text + "\n"


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _trace_content(value: Any) -> str:
    selected = str(value or "full").strip().lower()
    if selected not in {"full", "metadata"}:
        raise ValueError("trace_content must be 'full' or 'metadata'")
    return selected


def _reject_unknown(raw: dict[str, Any], contract: type, *, kind: str) -> None:
    unknown = sorted(set(raw) - set(contract.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown {kind} field(s): {', '.join(unknown)}")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"expected string or list, got {type(value).__name__}")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _positive_int(value: Any, *, kind: str) -> int | None:
    parsed = _optional_int(value)
    if parsed is not None and parsed < 1:
        raise ValueError(f"{kind} must be positive")
    return parsed


def _require_unique(values: list[str], *, kind: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {kind} id(s): {', '.join(sorted(duplicates))}")


def _dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping, got {type(value).__name__}")
    return dict(value)


def _workload_overrides(value: Any, preset_id: str) -> dict[str, dict[str, Any]]:
    values = _dict(value)
    allowed = {"n_tasks", "n_attempts", "n_concurrent"}
    result: dict[str, dict[str, Any]] = {}
    for key, item in values.items():
        settings = _dict(item)
        unknown = sorted(set(settings) - allowed)
        if unknown:
            raise ValueError(
                f"preset {preset_id} workload {key} has unknown override(s): "
                f"{', '.join(unknown)}"
            )
        result[str(key)] = {
            name: _positive_int(
                selected,
                kind=f"preset {preset_id} workload {key} {name}",
            )
            for name, selected in settings.items()
        }
    return result


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return list(value)


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]
    return value
