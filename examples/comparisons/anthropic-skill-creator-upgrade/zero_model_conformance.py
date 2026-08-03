"""Execute the exact upstream validators against a frozen boundary matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE = "a5bcdd7e58cdff48566bf876f0a72a2008dcefbc"
CANDIDATE = "1ed29a03dc852d30fa6ef2ca53a67dc2c2c2c563"
SCRIPT_ROOT = "skills/skill-creator/scripts"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _frontmatter(name: str, extra: str | None) -> str:
    lines = [
        "---",
        f"name: {name}",
        "description: Review immutable local evidence. Use when a maintainer needs a bounded result.",
    ]
    if extra is not None:
        lines.append(extra)
    lines.extend(("---", "", "# Evidence review", "", "Inspect local evidence."))
    return "\n".join(lines) + "\n"


def _run_validator(script: Path, root: Path) -> bool:
    env = {"PATH": os.environ.get("PATH", "")}
    python_path = os.environ.get("PYTHONPATH")
    if python_path:
        env["PYTHONPATH"] = python_path
    completed = subprocess.run(
        (sys.executable, script.as_posix(), root.as_posix()),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.returncode == 0


def _expected() -> dict[str, dict[str, object]]:
    return {
        "baseline": {
            "frontmatter": {
                "absent": True,
                "empty": False,
                "one_character": False,
                "length_500": False,
                "length_501": False,
                "non_string": False,
                "unknown_key": False,
            },
            "names": {"40": True, "41": True, "64": True, "65": False},
            "help_max_characters": 40,
        },
        "candidate": {
            "frontmatter": {
                "absent": True,
                "empty": True,
                "one_character": True,
                "length_500": True,
                "length_501": False,
                "non_string": False,
                "unknown_key": False,
            },
            "names": {"40": True, "41": True, "64": True, "65": False},
            "help_max_characters": 64,
        },
    }


def run_conformance(repo: Path) -> dict[str, object]:
    revisions = {"baseline": BASELINE, "candidate": CANDIDATE}
    observed: dict[str, dict[str, object]] = {}
    sources: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="fugue-skill-creator-conformance-") as raw:
        root = Path(raw)
        for role, commit in revisions.items():
            role_root = root / role
            role_root.mkdir()
            validator_source = _git(
                repo,
                "show",
                f"{commit}:{SCRIPT_ROOT}/quick_validate.py",
            )
            initializer_source = _git(
                repo,
                "show",
                f"{commit}:{SCRIPT_ROOT}/init_skill.py",
            )
            validator = role_root / "quick_validate.py"
            initializer = role_root / "init_skill.py"
            validator.write_text(validator_source, encoding="utf-8")
            initializer.write_text(initializer_source, encoding="utf-8")
            sources[role] = {
                "commit": commit,
                "validator_sha256": hashlib.sha256(
                    validator_source.encode()
                ).hexdigest(),
                "initializer_sha256": hashlib.sha256(
                    initializer_source.encode()
                ).hexdigest(),
            }

            values: dict[str, str | None] = {
                "absent": None,
                "empty": 'compatibility: ""',
                "one_character": "compatibility: x",
                "length_500": "compatibility: " + "x" * 500,
                "length_501": "compatibility: " + "x" * 501,
                "non_string": "compatibility: [Linux]",
                "unknown_key": "audience: maintainers",
            }
            frontmatter_results: dict[str, bool] = {}
            for case, extra in values.items():
                case_root = role_root / f"frontmatter-{case}"
                case_root.mkdir()
                (case_root / "SKILL.md").write_text(
                    _frontmatter("evidence-review", extra),
                    encoding="utf-8",
                )
                frontmatter_results[case] = _run_validator(validator, case_root)

            name_results: dict[str, bool] = {}
            for size in (40, 41, 64, 65):
                case_root = role_root / f"name-{size}"
                case_root.mkdir()
                (case_root / "SKILL.md").write_text(
                    _frontmatter("a" * size, None), encoding="utf-8"
                )
                name_results[str(size)] = _run_validator(validator, case_root)

            help_run = subprocess.run(
                (sys.executable, initializer.as_posix()),
                check=False,
                capture_output=True,
                text=True,
                env={"PATH": os.environ.get("PATH", "")},
            )
            help_output = help_run.stdout + help_run.stderr
            help_max = 64 if "Max 64 characters" in help_output else 40
            observed[role] = {
                "frontmatter": frontmatter_results,
                "names": name_results,
                "help_max_characters": help_max,
            }

    expected = _expected()
    return {
        "schema_version": 1,
        "id": "anthropic-skill-creator-zero-model-conformance-v1",
        "sources": sources,
        "expected": expected,
        "observed": observed,
        "status": "passed" if observed == expected else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anthropic-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_conformance(args.anthropic_repo.resolve())
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
