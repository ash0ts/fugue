"""Prepare immutable confirmatory task archives without network access.

The archives intentionally contain only task inputs.  In particular, they do
not contain a validator whose implementation reveals the expected answer.
Validation is performed independently by the locked host-side scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from prepare_sources import DECLARATION, _source_record
from zero_model_conformance import run_conformance

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = Path(__file__).resolve().parent
TASKS = EXAMPLE / "confirmatory-tasks.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".fugue/comparison-resources/anthropic-skill-creator-upgrade"
)

_BODY = """\
# {title}

## Workflow

1. Inspect only the supplied local inputs before acting.
2. Cite the local path for every factual observation.
3. Stop and report any missing requirement instead of guessing.

## Verification

Return a bounded maintainer result and preserve the existing package contract.
"""

_REFERENCE = """\
# Operations reference

Inputs are local and immutable. Do not install packages, contact the network,
or serialize credentials. Treat unavailable evidence as unavailable, not zero.
"""


def _skill(
    name: str,
    *,
    description: str | None = None,
    extra_frontmatter: tuple[str, ...] = (),
    body: str | None = None,
) -> str:
    title = " ".join(part.title() for part in name.split("-"))
    description = description or (
        f"Review local {title.lower()} inputs and produce a bounded result. "
        "Use when a maintainer needs an offline, evidence-grounded workflow."
    )
    frontmatter = (f"name: {name}", f"description: {description}") + tuple(
        extra_frontmatter
    )
    return "---\n" + "\n".join(frontmatter) + "\n---\n\n" + (
        body or _BODY.format(title=title)
    )


def _initial_files(task_id: str) -> dict[str, str]:
    cases: dict[str, dict[str, str]] = {
        "as-dev-preserve-runtime-metadata": {
            "skills/incident-evidence-review/SKILL.md": _skill(
                "incident-evidence-review",
                extra_frontmatter=(
                    "compatibility: macOS 13+ or Ubuntu 22.04+, Python 3.12+, and local read access",
                ),
            ),
            "skills/incident-evidence-review/references/operations.md": _REFERENCE,
        },
        "as-dev-repair-runtime-metadata": {
            "skills/offline-release-audit/SKILL.md": _skill(
                "offline-release-audit",
                extra_frontmatter=("compatibility: " + "x" * 501,),
            ),
            "requirements.md": (
                "# Deployment requirements\n\nLinux, jq 1.7+, and offline operation are required.\n"
            ),
        },
        "as-dev-repair-unknown-metadata": {
            "skills/maintainer-evidence-review/SKILL.md": _skill(
                "maintainer-evidence-review",
                extra_frontmatter=("audience: maintainers",),
            ),
        },
        "as-dev-init-help-diagnosis": {
            "scripts/init_skill.py": (
                "def help_text():\n"
                "    return 'Skill name requirements: lowercase kebab-case; Max 40 characters'\n"
            ),
            "scripts/name_policy.py": (
                "MAX_NAME_CHARACTERS = 64\n"
                "def valid_name(value):\n"
                "    return bool(value) and len(value) <= MAX_NAME_CHARACTERS\n"
            ),
        },
        "as-holdout-update-add-platform-requirements": {
            "skills/model-card-review/SKILL.md": _skill("model-card-review"),
            "deployment-requirements.md": (
                "# Deployment requirements\n\nUbuntu 24.04+, Python 3.12+, and read-only local access are required.\n"
            ),
        },
        "as-holdout-preserve-license-metadata": {
            "skills/archive-integrity-review/SKILL.md": _skill(
                "archive-integrity-review",
                extra_frontmatter=(
                    "license: Apache-2.0",
                    "compatibility: Linux with tar 1.34+ and offline inputs",
                ),
            ),
            "skills/archive-integrity-review/references/operations.md": _REFERENCE,
        },
        "as-holdout-preserve-allowed-tools": {
            "skills/read-only-log-review/SKILL.md": _skill(
                "read-only-log-review",
                extra_frontmatter=(
                    "allowed-tools: Read Grep Glob",
                    "compatibility: macOS or Linux with local read access",
                ),
            ),
            "skills/read-only-log-review/references/operations.md": _REFERENCE,
        },
        "as-holdout-existing-metadata-audit": {
            "skills/offline-checksum-review/SKILL.md": _skill(
                "offline-checksum-review",
                extra_frontmatter=(
                    "compatibility: Linux or macOS with sha256sum and offline inputs",
                ),
            ),
        },
        "as-holdout-runtime-metadata-repair": {
            "skills/fleet-evidence-review/SKILL.md": _skill(
                "fleet-evidence-review",
                extra_frontmatter=("compatibility: [Linux, Git 2.40+]",),
            ),
        },
        "as-holdout-metadata-boundary-alpha": {
            "skills/boundary-metadata-review/SKILL.md": _skill(
                "boundary-metadata-review",
                extra_frontmatter=("compatibility: " + "L" * 500,),
            ),
        },
        "as-holdout-metadata-boundary-beta": {
            "skills/overbound-metadata-review/SKILL.md": _skill(
                "overbound-metadata-review",
                extra_frontmatter=("compatibility: " + "L" * 501,),
            ),
            "requirements.md": (
                "# Runtime requirements\n\nLinux and offline local inputs are required.\n"
            ),
        },
        "as-holdout-repair-unknown-metadata": {
            "skills/change-evidence-review/SKILL.md": _skill(
                "change-evidence-review",
                extra_frontmatter=("audience: release-engineering",),
            ),
        },
    }
    return cases.get(task_id, {})


def _archive(path: Path, files: dict[str, str]) -> dict[str, object]:
    entries = {"README.md": "# Immutable task workspace\n\nWork only in this extracted tree.\n"}
    entries.update(files)
    with tarfile.open(path, "w") as archive:
        for relative, content in sorted(entries.items()):
            payload = content.encode()
            info = tarfile.TarInfo((Path("workspace") / relative).as_posix())
            info.size = len(payload)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return {
        "archive": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_count": len(entries),
        "paths_digest": hashlib.sha256(
            "\n".join(sorted(entries)).encode()
        ).hexdigest(),
    }


def _prepare_exact_sources(
    anthropic_repo: Path, output: Path
) -> dict[str, object]:
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    if declaration.get("repository") != "https://github.com/anthropics/skills":
        raise RuntimeError("unexpected Skill source repository")
    if declaration.get("path") != "skills/skill-creator":
        raise RuntimeError("unexpected Skill source path")
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        _source_record(
            anthropic_repo,
            output,
            role=role,
            declared=declaration[role],
            skill_path=str(declaration["path"]),
        )
        for role in ("baseline", "candidate")
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "repository": declaration["repository"],
        "path": declaration["path"],
        "sources": sources,
    }
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def prepare(anthropic_repo: Path, output: Path) -> dict[str, object]:
    source_manifest = _prepare_exact_sources(anthropic_repo, output)
    task_output = output / "confirmatory"
    task_output.mkdir(parents=True, exist_ok=True)
    task_rows = [
        json.loads(line)
        for line in TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    archives: list[dict[str, object]] = []
    for task in task_rows:
        task_id = str(task["id"])
        path = task_output / f"{task_id}.tar"
        record = _archive(path, _initial_files(task_id))
        record["task_id"] = task_id
        archives.append(record)

    conformance = run_conformance(anthropic_repo)
    if conformance["status"] != "passed":
        raise RuntimeError("zero-model conformance did not match the locked matrix")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "id": "anthropic-skill-creator-confirmatory-v1",
        "source_manifest_digest": source_manifest["manifest_digest"],
        "tasks": archives,
        "zero_model_conformance": conformance,
    }
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "confirmatory-preparation.lock.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anthropic-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.anthropic_repo.resolve(), args.output.resolve())
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
