from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml
from filelock import FileLock

SOURCE_POLICY_VERSION = "1"
SOURCE_CACHE_ROOT = Path(".fugue") / "cache" / "sources" / "v1"
SKILL_CACHE_ROOT = Path(".fugue") / "cache" / "skills" / "v1"
SKILL_REVIEW_ROOT = Path(".fugue") / "runtime" / "skill-reviews"
SKILL_SOURCE_ROOT = Path("configs") / "fugue" / "skill-sources"
SKILL_LIBRARY_ROOT = Path("configs") / "fugue" / "skills"
SKILL_LOCK_PATH = Path("configs") / "fugue" / "skills.lock.yaml"

MAX_SKILL_FILES = 500
MAX_SKILL_BYTES = 20 * 1024 * 1024
MAX_SKILL_FILE_BYTES = 5 * 1024 * 1024

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
_RASTER_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"RIFF",
)
_SCRIPT_SUFFIXES = {
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".ps1",
    ".cmd",
    ".bat",
    ".rb",
    ".pl",
}
_PACKAGE_MANIFESTS = {
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "poetry.lock",
    "cargo.toml",
    "go.mod",
    "gemfile",
}


class SkillSetupRequired(RuntimeError):
    """Raised when a remote skill has not been reviewed and locked."""


@dataclass(frozen=True)
class GitSourceSpec:
    type: str
    url: str
    ref: str
    path: str | None = None

    def __post_init__(self) -> None:
        if self.type != "git":
            raise ValueError("git source type must be 'git'")
        canonical_git_url(self.url)
        if not self.ref or not str(self.ref).strip():
            raise ValueError("git source ref is required")
        if self.path is not None:
            validate_relative_source_path(self.path)


@dataclass(frozen=True)
class SkillSourceSpec:
    id: str
    source: GitSourceSpec


@dataclass(frozen=True)
class SourceFile:
    path: str
    mode: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SourceFinding:
    id: str
    severity: str
    detail: str


@dataclass(frozen=True)
class SkillInspection:
    id: str
    source_url: str
    requested_ref: str
    resolved_commit: str
    source_path: str
    declared_name: str
    digest: str
    inventory_digest: str
    total_files: int
    total_bytes: int
    license_status: str
    files: tuple[SourceFile, ...]
    findings: tuple[SourceFinding, ...] = ()
    policy_version: str = SOURCE_POLICY_VERSION

    @property
    def acknowledgement_ids(self) -> tuple[str, ...]:
        return tuple(
            item.id for item in self.findings if item.severity == "acknowledge"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillLockEntry:
    id: str
    source_url: str
    requested_ref: str
    resolved_commit: str
    source_path: str
    declared_name: str
    digest: str
    inventory_digest: str
    total_files: int
    total_bytes: int
    license_status: str
    findings: tuple[dict[str, str], ...] = ()
    approved_findings: tuple[str, ...] = ()
    policy_version: str = SOURCE_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["findings"] = list(self.findings)
        value["approved_findings"] = list(self.approved_findings)
        return value


@dataclass(frozen=True)
class ResolvedSkill:
    id: str
    declared_name: str
    path: Path
    digest: str
    source_url: str | None = None
    requested_ref: str | None = None
    resolved_commit: str | None = None
    source_path: str | None = None
    license_status: str | None = None
    policy_version: str | None = None
    findings: tuple[dict[str, str], ...] = ()

    def provenance(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "id": self.id,
                "declared_name": self.declared_name,
                "digest": self.digest,
                "source_url": self.source_url,
                "requested_ref": self.requested_ref,
                "resolved_commit": self.resolved_commit,
                "source_path": self.source_path,
                "license_status": self.license_status,
                "policy_version": self.policy_version,
                "findings": list(self.findings),
            }.items()
            if value not in (None, [], (), {})
        }


def canonical_git_url(value: str) -> str:
    parsed = urlparse(str(value).strip())
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("V1 git sources must use public https://github.com URLs")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("git source URLs may not contain credentials, queries, or fragments")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ValueError("git source URL must identify exactly one owner/repository")
    owner, repository = parts
    repository = repository.removesuffix(".git")
    if not _ID_RE.fullmatch(owner) or not _ID_RE.fullmatch(repository):
        raise ValueError("git source owner and repository contain unsupported characters")
    return f"https://github.com/{owner}/{repository}"


def validate_relative_source_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("source paths must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("source paths must be non-empty relative paths without '.' or '..'")
    return path.as_posix()


def load_skill_source(skill_id: str, repo_root: Path) -> SkillSourceSpec:
    _validate_id(skill_id, "skill source id")
    path = repo_root / SKILL_SOURCE_ROOT / f"{skill_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"skill source not found: {skill_id}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: skill source must be a mapping")
    unknown = sorted(set(raw) - {"id", "source"})
    if unknown:
        raise ValueError(f"{path}: unknown skill source field(s): {', '.join(unknown)}")
    declared_id = str(raw.get("id") or skill_id)
    if declared_id != skill_id:
        raise ValueError(f"{path}: declared id {declared_id!r} does not match filename")
    source_raw = raw.get("source") or {}
    if not isinstance(source_raw, dict):
        raise ValueError(f"{path}: source must be a mapping")
    unknown_source = sorted(set(source_raw) - {"type", "url", "ref", "path"})
    if unknown_source:
        raise ValueError(
            f"{path}: unknown git source field(s): {', '.join(unknown_source)}"
        )
    source = GitSourceSpec(
        type=str(source_raw.get("type") or ""),
        url=str(source_raw.get("url") or ""),
        ref=str(source_raw.get("ref") or ""),
        path=(str(source_raw["path"]) if source_raw.get("path") else None),
    )
    if source.path is None:
        raise ValueError(f"{path}: remote skill sources require an explicit path")
    if not _SHA_RE.fullmatch(source.ref):
        raise ValueError(f"{path}: remote skill sources require a full commit SHA")
    return SkillSourceSpec(id=skill_id, source=source)


def list_skill_source_ids(repo_root: Path) -> list[str]:
    root = repo_root / SKILL_SOURCE_ROOT
    if not root.is_dir():
        return []
    return sorted(path.stem for path in root.glob("*.yaml") if path.is_file())


def prepare_skill_source(
    skill_id: str,
    repo_root: Path,
    *,
    refresh: bool = False,
) -> SkillInspection:
    _validate_id(skill_id, "skill source id")
    lock_path = repo_root / SOURCE_CACHE_ROOT / "locks" / f"{skill_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path.as_posix()):
        return _prepare_skill_source(skill_id, repo_root, refresh=refresh)


def _prepare_skill_source(
    skill_id: str,
    repo_root: Path,
    *,
    refresh: bool,
) -> SkillInspection:
    spec = load_skill_source(skill_id, repo_root)
    source = spec.source
    canonical = canonical_git_url(source.url)
    bare = _bare_repo_path(repo_root, canonical)
    bare.parent.mkdir(parents=True, exist_ok=True)
    if not bare.exists():
        _run_git(["init", "--bare", "--quiet", bare.as_posix()], cwd=repo_root)
    fetched = refresh or not _has_commit(bare, source.ref)
    if fetched:
        _run_git(
            [
                "-C",
                bare.as_posix(),
                "fetch",
                "--quiet",
                "--depth=1",
                f"{canonical}.git",
                source.ref,
            ],
            cwd=repo_root,
            timeout=120,
        )
    resolved_ref = "FETCH_HEAD" if fetched else source.ref
    commit = _git_output(
        ["-C", bare.as_posix(), "rev-parse", f"{resolved_ref}^{{commit}}"],
        cwd=repo_root,
    ).strip()
    if not _SHA_RE.fullmatch(commit):
        raise ValueError(f"could not resolve {canonical}@{source.ref} to a commit")
    if commit != source.ref:
        raise ValueError(f"{canonical}@{source.ref} does not identify a commit directly")
    inspection, blobs = _inspect_skill_tree(spec, bare, commit)
    _materialize_skill(repo_root, inspection, blobs)
    review_path = _review_path(repo_root, skill_id)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(yaml.safe_dump(inspection.to_dict(), sort_keys=False))
    return inspection


def approve_skill_source(
    skill_id: str,
    digest: str,
    repo_root: Path,
    *,
    acknowledged_findings: tuple[str, ...] = (),
) -> SkillLockEntry:
    _validate_id(skill_id, "skill source id")
    lock_path = repo_root / SOURCE_CACHE_ROOT / "locks" / f"{skill_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path.as_posix()):
        return _approve_skill_source(
            skill_id,
            digest,
            repo_root,
            acknowledged_findings=acknowledged_findings,
        )


def _approve_skill_source(
    skill_id: str,
    digest: str,
    repo_root: Path,
    *,
    acknowledged_findings: tuple[str, ...],
) -> SkillLockEntry:
    review_path = _review_path(repo_root, skill_id)
    if not review_path.is_file():
        raise SkillSetupRequired(
            f"skill {skill_id} has no review; run `fugue setup --skills` first"
        )
    raw = yaml.safe_load(review_path.read_text()) or {}
    inspection = _inspection_from_dict(raw)
    if inspection.id != skill_id:
        raise ValueError(
            f"skill review id mismatch: expected {skill_id!r}, got {inspection.id!r}"
        )
    spec = load_skill_source(skill_id, repo_root)
    expected_source = (
        canonical_git_url(spec.source.url),
        spec.source.ref,
        str(spec.source.path),
        SOURCE_POLICY_VERSION,
    )
    reviewed_source = (
        inspection.source_url,
        inspection.requested_ref,
        inspection.source_path,
        inspection.policy_version,
    )
    if reviewed_source != expected_source or not _SHA_RE.fullmatch(
        inspection.resolved_commit
    ):
        raise ValueError(
            f"skill {skill_id} review no longer matches its source declaration"
        )
    bare = _bare_repo_path(repo_root, inspection.source_url)
    if not _has_commit(bare, inspection.resolved_commit):
        raise SkillSetupRequired(
            f"skill {skill_id} reviewed Git object is missing; rerun `fugue setup --skills`"
        )
    verified, _ = _inspect_skill_tree(spec, bare, inspection.resolved_commit)
    if verified != inspection:
        raise ValueError(
            f"skill {skill_id} review no longer matches the inspected Git objects"
        )
    if inspection.digest != digest:
        raise ValueError(
            f"skill {skill_id} approval digest mismatch: expected {inspection.digest}"
        )
    cache_path = skill_cache_path(
        repo_root, inspection.digest, inspection.declared_name
    )
    actual_digest, actual_name = digest_local_skill(
        cache_path, fallback_name=inspection.declared_name
    )
    if (actual_digest, actual_name) != (
        inspection.digest,
        inspection.declared_name,
    ):
        raise ValueError(f"skill {skill_id} reviewed cache no longer matches its digest")
    required = set(inspection.acknowledgement_ids)
    acknowledged = set(acknowledged_findings)
    missing = sorted(required - acknowledged)
    if missing:
        raise ValueError(
            "skill approval requires acknowledgement of: " + ", ".join(missing)
        )
    entry = SkillLockEntry(
        id=inspection.id,
        source_url=inspection.source_url,
        requested_ref=inspection.requested_ref,
        resolved_commit=inspection.resolved_commit,
        source_path=inspection.source_path,
        declared_name=inspection.declared_name,
        digest=inspection.digest,
        inventory_digest=inspection.inventory_digest,
        total_files=inspection.total_files,
        total_bytes=inspection.total_bytes,
        license_status=inspection.license_status,
        findings=tuple(asdict(item) for item in inspection.findings),
        approved_findings=tuple(sorted(acknowledged)),
    )
    lock = load_skill_lock(repo_root)
    lock[skill_id] = entry
    _write_skill_lock(repo_root, lock)
    return entry


def load_skill_lock(repo_root: Path) -> dict[str, SkillLockEntry]:
    path = repo_root / SKILL_LOCK_PATH
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError(f"{path}: unsupported skill lock format")
    values = raw.get("skills") or {}
    if not isinstance(values, dict):
        raise ValueError(f"{path}: skills must be a mapping")
    result: dict[str, SkillLockEntry] = {}
    for skill_id, item in values.items():
        skill_id = str(skill_id)
        _validate_id(skill_id, "skill lock id")
        if not isinstance(item, dict):
            raise ValueError(f"{path}: lock entry {skill_id} must be a mapping")
        unknown = sorted(
            set(item)
            - {
                "id",
                "source_url",
                "requested_ref",
                "resolved_commit",
                "source_path",
                "declared_name",
                "digest",
                "inventory_digest",
                "total_files",
                "total_bytes",
                "license_status",
                "findings",
                "approved_findings",
                "policy_version",
            }
        )
        if unknown:
            raise ValueError(
                f"{path}: unknown lock entry field(s): {', '.join(unknown)}"
            )
        entry_id = str(item.get("id") or skill_id)
        if entry_id != skill_id:
            raise ValueError(
                f"{path}: lock entry id {entry_id!r} does not match {skill_id!r}"
            )
        source_url = str(item["source_url"])
        if canonical_git_url(source_url) != source_url:
            raise ValueError(f"{path}: lock entry {skill_id} source URL is not canonical")
        requested_ref = str(item["requested_ref"])
        resolved_commit = str(item["resolved_commit"])
        if not _SHA_RE.fullmatch(requested_ref) or resolved_commit != requested_ref:
            raise ValueError(
                f"{path}: lock entry {skill_id} must identify one pinned commit"
            )
        source_path = validate_relative_source_path(str(item["source_path"]))
        declared_name = _validate_id(str(item["declared_name"]), "skill name")
        digest = str(item["digest"])
        inventory_digest = str(item["inventory_digest"])
        if not _DIGEST_RE.fullmatch(digest) or not _DIGEST_RE.fullmatch(
            inventory_digest
        ):
            raise ValueError(f"{path}: lock entry {skill_id} has an invalid digest")
        total_files = _bounded_lock_integer(
            item["total_files"], maximum=MAX_SKILL_FILES, field="total_files", path=path
        )
        total_bytes = _bounded_lock_integer(
            item["total_bytes"], maximum=MAX_SKILL_BYTES, field="total_bytes", path=path
        )
        result[skill_id] = SkillLockEntry(
            id=entry_id,
            source_url=source_url,
            requested_ref=requested_ref,
            resolved_commit=resolved_commit,
            source_path=source_path,
            declared_name=declared_name,
            digest=digest,
            inventory_digest=inventory_digest,
            total_files=total_files,
            total_bytes=total_bytes,
            license_status=str(item["license_status"]),
            findings=tuple(dict(value) for value in item.get("findings") or []),
            approved_findings=tuple(str(value) for value in item.get("approved_findings") or []),
            policy_version=str(item.get("policy_version") or ""),
        )
    return result


def resolve_skill(skill_id: str, repo_root: Path) -> ResolvedSkill:
    _validate_id(skill_id, "skill id")
    local = repo_root / SKILL_LIBRARY_ROOT / skill_id
    if (local / "SKILL.md").is_file():
        digest, declared_name = digest_local_skill(local, fallback_name=skill_id)
        return ResolvedSkill(
            id=skill_id,
            declared_name=declared_name,
            path=local,
            digest=digest,
            license_status="project-owned",
        )
    spec = load_skill_source(skill_id, repo_root)
    entry = load_skill_lock(repo_root).get(skill_id)
    if entry is None:
        raise SkillSetupRequired(
            f"remote skill {skill_id} is not approved; run `fugue setup --skills`"
        )
    expected = (
        canonical_git_url(spec.source.url),
        spec.source.ref,
        str(spec.source.path),
        SOURCE_POLICY_VERSION,
    )
    current = (
        entry.source_url,
        entry.requested_ref,
        entry.source_path,
        entry.policy_version,
    )
    if current != expected:
        raise SkillSetupRequired(
            f"remote skill {skill_id} lock is stale; rerun `fugue setup --skills`"
        )
    path = skill_cache_path(repo_root, entry.digest, entry.declared_name)
    if not (path / "SKILL.md").is_file():
        raise SkillSetupRequired(
            f"remote skill {skill_id} cache is missing; rerun `fugue setup --skills`"
        )
    actual, declared_name = digest_local_skill(path, fallback_name=entry.declared_name)
    if actual != entry.digest or declared_name != entry.declared_name:
        raise SkillSetupRequired(
            f"remote skill {skill_id} cache does not match its approved digest"
        )
    return ResolvedSkill(
        id=skill_id,
        declared_name=entry.declared_name,
        path=path,
        digest=entry.digest,
        source_url=entry.source_url,
        requested_ref=entry.requested_ref,
        resolved_commit=entry.resolved_commit,
        source_path=entry.source_path,
        license_status=entry.license_status,
        policy_version=entry.policy_version,
        findings=entry.findings,
    )


def resolve_skills(skill_ids: list[str], repo_root: Path) -> list[ResolvedSkill]:
    resolved = [resolve_skill(skill_id, repo_root) for skill_id in skill_ids]
    names: dict[str, list[str]] = {}
    for item in resolved:
        names.setdefault(item.declared_name, []).append(item.id)
    duplicates = {name: ids for name, ids in names.items() if len(ids) > 1}
    if duplicates:
        detail = "; ".join(f"{name}: {', '.join(ids)}" for name, ids in sorted(duplicates.items()))
        raise ValueError(f"duplicate injected skill name(s): {detail}")
    return resolved


def digest_local_skill(path: Path, *, fallback_name: str) -> tuple[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"skill path must be a real directory: {path}")
    skill_file = path / "SKILL.md"
    if not skill_file.is_file() or skill_file.is_symlink():
        raise ValueError(f"skill path must contain a regular SKILL.md: {path}")
    hasher = hashlib.sha256()
    count = 0
    total = 0
    file_paths = sorted(
        path.rglob("*"),
        key=lambda file_path: file_path.relative_to(path).as_posix(),
    )
    for file_path in file_paths:
        if file_path.is_symlink():
            raise ValueError(f"skill bundles may not contain symlinks: {file_path}")
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(path).as_posix()
        content = file_path.read_bytes()
        count += 1
        total += len(content)
        if count > MAX_SKILL_FILES or total > MAX_SKILL_BYTES or len(content) > MAX_SKILL_FILE_BYTES:
            raise ValueError(f"skill bundle exceeds safety limits: {path}")
        mode = "100755" if os.access(file_path, os.X_OK) else "100644"
        _update_bundle_hash(hasher, relative, mode, content)
    declared_name = _skill_name(skill_file.read_text(), fallback=fallback_name)
    return f"sha256:{hasher.hexdigest()}", declared_name


def skill_cache_path(repo_root: Path, digest: str, declared_name: str) -> Path:
    if not _DIGEST_RE.fullmatch(digest):
        raise ValueError(f"invalid skill digest: {digest!r}")
    _validate_id(declared_name, "skill name")
    return repo_root / SKILL_CACHE_ROOT / digest.removeprefix("sha256:") / declared_name


def _bounded_lock_integer(
    value: Any, *, maximum: int, field: str, path: Path
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{path}: lock entry {field} is outside the safety limit")
    return value


def _inspect_skill_tree(
    spec: SkillSourceSpec,
    bare: Path,
    commit: str,
) -> tuple[SkillInspection, dict[str, bytes]]:
    source_path = validate_relative_source_path(str(spec.source.path))
    tree = _git_output_bytes(
        [
            "-C",
            bare.as_posix(),
            "ls-tree",
            "-r",
            "-z",
            "-l",
            commit,
            "--",
            source_path,
        ],
        cwd=bare.parent,
    )
    records = [record for record in tree.split(b"\0") if record]
    if not records:
        raise ValueError(f"remote skill path does not exist: {source_path}")
    if len(records) > MAX_SKILL_FILES:
        raise ValueError(f"remote skill exceeds {MAX_SKILL_FILES} files")
    files: list[SourceFile] = []
    blobs: dict[str, bytes] = {}
    total = 0
    text_parts: list[tuple[str, str]] = []
    prefix = source_path.rstrip("/") + "/"
    for record in records:
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id, raw_size = metadata.decode().split(" ", 3)
        full_path = raw_path.decode("utf-8")
        if not full_path.startswith(prefix):
            raise ValueError(f"git tree escaped selected skill path: {full_path}")
        relative = full_path[len(prefix) :]
        validate_relative_source_path(relative)
        if object_type != "blob" or mode not in {"100644", "100755"}:
            kind = "submodule" if mode == "160000" else "symlink or special file"
            raise ValueError(f"remote skill contains unsupported {kind}: {full_path}")
        size = int(raw_size)
        if size > MAX_SKILL_FILE_BYTES:
            raise ValueError(f"remote skill file exceeds size limit: {full_path}")
        total += size
        if total > MAX_SKILL_BYTES:
            raise ValueError(f"remote skill exceeds {MAX_SKILL_BYTES} bytes")
        content = _git_output_bytes(
            ["-C", bare.as_posix(), "cat-file", "blob", object_id], cwd=bare.parent
        )
        if len(content) != size:
            raise ValueError(f"git blob size changed while inspecting {full_path}")
        if content.startswith(_LFS_PREFIX):
            raise ValueError(f"Git LFS pointers are not supported: {full_path}")
        text = _decode_allowed_content(relative, content)
        if text is not None:
            text_parts.append((relative, text))
        digest = hashlib.sha256(content).hexdigest()
        files.append(SourceFile(path=relative, mode=mode, size=size, sha256=digest))
        blobs[relative] = content
    bundle_hash = hashlib.sha256()
    inventory_hash = hashlib.sha256()
    for item in sorted(files, key=lambda value: value.path):
        _update_bundle_hash(bundle_hash, item.path, item.mode, blobs[item.path])
        inventory_hash.update(
            f"{item.path}\0{item.mode}\0{item.size}\0{item.sha256}\0".encode()
        )
    skill_content = blobs.get("SKILL.md")
    if skill_content is None:
        raise ValueError(f"remote skill path lacks SKILL.md: {source_path}")
    skill_text = skill_content.decode("utf-8")
    declared_name = _skill_name(skill_text, fallback="")
    if not declared_name:
        raise ValueError("remote SKILL.md requires a frontmatter name")
    findings = _scan_findings(files, text_parts)
    root_license = _root_license_status(bare, commit)
    frontmatter_license = _skill_frontmatter(skill_text).get("license")
    if root_license == "missing" and frontmatter_license:
        license_status = "declared-only"
        findings.append(
            SourceFinding(
                id="missing-license-file",
                severity="acknowledge",
                detail=f"SKILL.md declares {frontmatter_license!s} but the repository has no root LICENSE file",
            )
        )
    elif root_license == "missing":
        license_status = "missing"
        findings.append(
            SourceFinding(
                id="missing-license-file",
                severity="acknowledge",
                detail="repository has no root LICENSE file",
            )
        )
    else:
        license_status = str(frontmatter_license or root_license)
    if _has_excluded_plugin_content(bare, commit, source_path):
        findings.append(
            SourceFinding(
                id="plugin-content-excluded",
                severity="info",
                detail="plugin manifests, hooks, or extensions outside the selected skill path are excluded",
            )
        )
    inspection = SkillInspection(
        id=spec.id,
        source_url=canonical_git_url(spec.source.url),
        requested_ref=spec.source.ref,
        resolved_commit=commit,
        source_path=source_path,
        declared_name=declared_name,
        digest=f"sha256:{bundle_hash.hexdigest()}",
        inventory_digest=f"sha256:{inventory_hash.hexdigest()}",
        total_files=len(files),
        total_bytes=total,
        license_status=license_status,
        files=tuple(files),
        findings=tuple(_dedupe_findings(findings)),
    )
    return inspection, blobs


def _materialize_skill(
    repo_root: Path,
    inspection: SkillInspection,
    blobs: dict[str, bytes],
) -> None:
    destination = skill_cache_path(repo_root, inspection.digest, inspection.declared_name)
    cache_lock = destination.parent / ".materialize.lock"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(cache_lock.as_posix()):
        if (destination / "SKILL.md").is_file():
            try:
                current = digest_local_skill(
                    destination, fallback_name=inspection.declared_name
                )
            except ValueError:
                current = ("", "")
            if current == (inspection.digest, inspection.declared_name):
                return
            _remove_tree(destination)
        elif destination.exists():
            if destination.is_dir():
                _remove_tree(destination)
            else:
                destination.unlink()
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{inspection.declared_name}.", dir=destination.parent
            )
        )
        try:
            modes = {item.path: item.mode for item in inspection.files}
            for relative, content in blobs.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(0o555 if modes[relative] == "100755" else 0o444)
            os.replace(staging, destination)
        except Exception:
            _remove_tree(staging)
            raise


def _scan_findings(
    files: list[SourceFile], text_parts: list[tuple[str, str]]
) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    lower_names = {PurePosixPath(item.path).name.lower() for item in files}
    executable = [
        item.path
        for item in files
        if item.mode == "100755" or PurePosixPath(item.path).suffix.lower() in _SCRIPT_SUFFIXES
    ]
    if executable:
        findings.append(
            SourceFinding(
                "executable-files",
                "acknowledge",
                "bundle contains executable or script files: " + ", ".join(executable[:8]),
            )
        )
    manifests = sorted(lower_names & _PACKAGE_MANIFESTS)
    if manifests:
        findings.append(
            SourceFinding(
                "package-manifest",
                "acknowledge",
                "bundle contains package manifests: " + ", ".join(manifests),
            )
        )
    combined = "\n".join(text for _, text in text_parts)
    patterns = (
        ("network-access", r"https?://|\bcurl\b|\bwget\b|\bwebfetch\b", "network URLs or fetch commands"),
        ("telemetry", r"telemetry|analytics|tracking pixel", "telemetry-related instructions"),
        ("credential-access", r"api[_ -]?key|credential|secret|\.env\b|~/|\$HOME", "credential or home-directory access"),
        ("destructive-commands", r"\brm\s+-[a-zA-Z]*r|\bsudo\b|git\s+reset\s+--hard", "destructive or privileged commands"),
        ("repository-mutation", r"git\s+(commit|push|worktree|checkout)|write_text\(|mkdir\(|create or update", "repository mutation instructions"),
        ("server-listener", r"localhost|127\.0\.0\.1|listen\(|http\.server|start-server|server\.cjs", "server or listener behavior"),
        ("subagent-control", r"subagent|dispatching-parallel|task tool", "subagent or task orchestration"),
    )
    for finding_id, pattern, detail in patterns:
        if re.search(pattern, combined, re.IGNORECASE):
            findings.append(SourceFinding(finding_id, "acknowledge", detail))
    return findings


def _decode_allowed_content(path: str, content: bytes) -> str | None:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        lower = path.lower()
        if lower.endswith((".png", ".jpg", ".jpeg")) and content.startswith(_RASTER_MAGIC[:2]):
            return None
        if lower.endswith(".webp") and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
            return None
        raise ValueError(f"unsupported binary content in skill bundle: {path}") from None


def _skill_frontmatter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        raise ValueError("SKILL.md frontmatter is not terminated") from None
    raw = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(raw, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    return raw


def _skill_name(text: str, *, fallback: str) -> str:
    name = str(_skill_frontmatter(text).get("name") or fallback).strip()
    if name and not _ID_RE.fullmatch(name):
        raise ValueError(f"invalid SKILL.md name: {name!r}")
    return name


def _root_license_status(bare: Path, commit: str) -> str:
    output = _git_output(
        ["-C", bare.as_posix(), "ls-tree", "--name-only", commit], cwd=bare.parent
    )
    candidates = [
        line.strip()
        for line in output.splitlines()
        if line.strip().lower() in {"license", "license.md", "license.txt", "copying"}
    ]
    return candidates[0] if candidates else "missing"


def _has_excluded_plugin_content(bare: Path, commit: str, source_path: str) -> bool:
    output = _git_output(
        ["-C", bare.as_posix(), "ls-tree", "-r", "--name-only", commit], cwd=bare.parent
    )
    markers = (".claude-plugin/", ".codex-plugin/", ".cursor-plugin/", "hooks/", "extensions/")
    return any(
        line.startswith(markers) and not line.startswith(source_path.rstrip("/") + "/")
        for line in output.splitlines()
    )


def _write_skill_lock(repo_root: Path, values: dict[str, SkillLockEntry]) -> None:
    path = repo_root / SKILL_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "skills": {key: values[key].to_dict() for key in sorted(values)},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _inspection_from_dict(raw: dict[str, Any]) -> SkillInspection:
    return SkillInspection(
        id=str(raw["id"]),
        source_url=str(raw["source_url"]),
        requested_ref=str(raw["requested_ref"]),
        resolved_commit=str(raw["resolved_commit"]),
        source_path=str(raw["source_path"]),
        declared_name=str(raw["declared_name"]),
        digest=str(raw["digest"]),
        inventory_digest=str(raw["inventory_digest"]),
        total_files=int(raw["total_files"]),
        total_bytes=int(raw["total_bytes"]),
        license_status=str(raw["license_status"]),
        files=tuple(SourceFile(**item) for item in raw.get("files") or []),
        findings=tuple(SourceFinding(**item) for item in raw.get("findings") or []),
        policy_version=str(raw.get("policy_version") or ""),
    )


def _review_path(repo_root: Path, skill_id: str) -> Path:
    return repo_root / SKILL_REVIEW_ROOT / f"{skill_id}.yaml"


def _bare_repo_path(repo_root: Path, canonical_url: str) -> Path:
    digest = hashlib.sha256(canonical_url.encode()).hexdigest()
    return repo_root / SOURCE_CACHE_ROOT / "git" / f"{digest}.git"


def _has_commit(bare: Path, ref: str) -> bool:
    if not bare.is_dir() or not _SHA_RE.fullmatch(ref):
        return False
    result = _run_git(
        ["-C", bare.as_posix(), "cat-file", "-e", f"{ref}^{{commit}}"],
        cwd=bare.parent,
        check=False,
    )
    return result.returncode == 0


def _git_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "GIT_ASKPASS",
            "GIT_DIR",
            "GIT_INDEX_FILE",
            "GIT_PROXY_COMMAND",
            "GIT_SSH_COMMAND",
            "GIT_WORK_TREE",
            "SSH_ASKPASS",
        }
    }
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return env


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 30,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=never",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=cwd,
        env=_git_env(),
        capture_output=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git source operation failed: {detail}")
    return result


def _git_output(args: list[str], *, cwd: Path) -> str:
    return _run_git(args, cwd=cwd).stdout.decode("utf-8")


def _git_output_bytes(args: list[str], *, cwd: Path) -> bytes:
    return _run_git(args, cwd=cwd).stdout


def _update_bundle_hash(
    hasher: Any, relative: str, mode: str, content: bytes
) -> None:
    hasher.update(relative.encode())
    hasher.update(b"\0")
    hasher.update(mode.encode())
    hasher.update(b"\0")
    hasher.update(hashlib.sha256(content).hexdigest().encode())
    hasher.update(b"\0")


def _dedupe_findings(values: list[SourceFinding]) -> list[SourceFinding]:
    result: list[SourceFinding] = []
    seen: set[str] = set()
    for item in values:
        if item.id not in seen:
            result.append(item)
            seen.add(item.id)
    return result


def _validate_id(value: str, kind: str) -> str:
    if not _ID_RE.fullmatch(str(value)):
        raise ValueError(f"invalid {kind}: {value!r}")
    return str(value)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        try:
            item.chmod(stat.S_IRWXU)
        except OSError:
            pass
    path.chmod(stat.S_IRWXU)
    shutil.rmtree(path)
