from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
class FeatureVariant:
    id: str
    label: str
    prompt_id: str | None = None
    skill_ids: list[str] = field(default_factory=list)
    memory: str | None = None
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
class ExperimentSpec:
    id: str
    title: str
    description: str = ""
    manifest: Path = Path("datasets/pilot.yaml")
    model: str | None = None
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
    debug: bool = False
    quiet: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest"] = self.manifest.as_posix()
        if self.jobs_dir is not None:
            data["jobs_dir"] = self.jobs_dir.as_posix()
        return data


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
    return experiment_from_yaml(path.read_text(), item_id=item_id)


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
    experiment_id = validate_id(
        raw.get("id") or item_id or "experiment", kind="experiment id"
    )
    variants = _variants(raw)
    if not variants:
        variants = [FeatureVariant(id="baseline", label="Baseline", memory="none")]
    return ExperimentSpec(
        id=experiment_id,
        title=str(raw.get("title") or experiment_id),
        description=str(raw.get("description") or ""),
        manifest=Path(str(raw.get("manifest") or "datasets/pilot.yaml")),
        model=_optional_str(raw.get("model")),
        run_name=_optional_str(raw.get("run_name")),
        tags=_string_list(raw.get("tags")),
        harnesses=_string_list(raw.get("harnesses")),
        variants=variants,
        n_attempts=_optional_int(raw.get("n_attempts") or raw.get("k")),
        n_concurrent=_optional_int(raw.get("n_concurrent")),
        n_tasks=_optional_int(raw.get("n_tasks")),
        jobs_dir=Path(str(raw["jobs_dir"])) if raw.get("jobs_dir") else None,
        environment=_dict(raw.get("environment")),
        artifacts=_list(raw.get("artifacts")),
        verifier=_dict(raw.get("verifier")),
        retry=_dict(raw.get("retry")),
        agent_kwargs=_dict(raw.get("agent_kwargs")),
        agent_env={str(k): str(v) for k, v in _dict(raw.get("agent_env")).items()},
        mcp_servers=[_dict(item) for item in _list(raw.get("mcp_servers"))],
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
    variant_id = validate_id(raw.get("id") or f"variant-{index}", kind="variant id")
    return FeatureVariant(
        id=variant_id,
        label=str(raw.get("label") or variant_id),
        prompt_id=_optional_str(raw.get("prompt_id")),
        skill_ids=_string_list(raw.get("skill_ids")),
        memory=_optional_str(raw.get("memory")) or "none",
        agent_kwargs=_dict(raw.get("agent_kwargs")),
        agent_env={str(k): str(v) for k, v in _dict(raw.get("agent_env")).items()},
        mcp_servers=[_dict(item) for item in _list(raw.get("mcp_servers"))],
        environment=_dict(raw.get("environment")),
        verifier=_dict(raw.get("verifier")),
        retry=_dict(raw.get("retry")),
        artifacts=_list(raw.get("artifacts")),
        enabled=bool(raw.get("enabled", True)),
    )


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


def _dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping, got {type(value).__name__}")
    return dict(value)


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return list(value)
