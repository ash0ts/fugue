from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Mapping
from typing import Any


def context_registration_digest(
    *,
    context_system_id: str,
    delivery: str,
    context_config_hash: str,
    command: str | None,
    servers: list[dict[str, Any]],
) -> str:
    payload = {
        "context_system_id": context_system_id,
        "delivery": delivery,
        "context_config_hash": context_config_hash,
        "command": command,
        "servers": sorted(servers, key=lambda item: str(item.get("name") or "")),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def skill_registration_probe_command(
    directory: str,
    assigned: list[str],
    registration_names: Mapping[str, str] | None = None,
) -> str:
    assigned = sorted(set(assigned))
    expected = {
        skill_id: str((registration_names or {}).get(skill_id) or skill_id)
        for skill_id in assigned
    }
    script = (
        "import hashlib,json,os,sys;"
        "from pathlib import Path;"
        "root=Path(os.path.expandvars(os.path.expanduser(sys.argv[1])));"
        "assigned=sorted(set(json.loads(sys.argv[2])));"
        "expected=json.loads(sys.argv[3]);"
        "files=sorted(root.rglob('SKILL.md')) if root.is_dir() else [];"
        "registered_names=sorted({"
        "(path.relative_to(root).parts[0] if len(path.relative_to(root).parts)>1 "
        "else '.') for path in files});"
        "registered=sorted(skill_id for skill_id,name in expected.items() "
        "if name in registered_names);"
        "missing=sorted(set(assigned)-set(registered));"
        "unexpected=sorted(set(registered_names)-set(expected.values()));"
        "ambiguous=sorted({name for name in expected.values() "
        "if list(expected.values()).count(name)>1});"
        "digest=hashlib.sha256();"
        "[(digest.update(path.relative_to(root).as_posix().encode()+b'\\0'),"
        "digest.update(path.read_bytes())) for path in files];"
        "payload={'skills_assigned':assigned,'skills_registered':registered,"
        "'registered_skill_names':registered_names,"
        "'missing_skills':missing,'unexpected_skills':unexpected,"
        "'ambiguous_skill_names':ambiguous,"
        "'directory':str(root),"
        "'skill_files':[path.relative_to(root).as_posix() for path in files],"
        "'registration_digest':('sha256:'+digest.hexdigest()) if files else None};"
        "print(json.dumps(payload,sort_keys=True));"
        "sys.exit(0 if not missing and not unexpected and not ambiguous else 2)"
    )
    return (
        f"python3 -c {shlex.quote(script)} {shlex.quote(directory)} "
        f"{shlex.quote(json.dumps(assigned))} {shlex.quote(json.dumps(expected))}"
    )
