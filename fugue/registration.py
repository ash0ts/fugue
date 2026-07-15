from __future__ import annotations

import hashlib
import json
import shlex
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


def skill_registration_probe_command(directory: str, assigned: list[str]) -> str:
    script = (
        "import hashlib,json,sys;"
        "from pathlib import Path;"
        "root=Path(sys.argv[1]);"
        "assigned=json.loads(sys.argv[2]);"
        "files=sorted(root.rglob('SKILL.md')) if root.is_dir() else [];"
        "registered=sorted({"
        "(path.relative_to(root).parts[0] if len(path.relative_to(root).parts)>1 "
        "else '.') for path in files});"
        "digest=hashlib.sha256();"
        "[(digest.update(path.relative_to(root).as_posix().encode()+b'\\0'),"
        "digest.update(path.read_bytes())) for path in files];"
        "payload={'skills_assigned':assigned,'skills_registered':registered,"
        "'skill_files':[path.relative_to(root).as_posix() for path in files],"
        "'registration_digest':('sha256:'+digest.hexdigest()) if files else None};"
        "print(json.dumps(payload,sort_keys=True));"
        "sys.exit(0 if len(registered)==len(assigned) else 2)"
    )
    return (
        f"python3 -c {shlex.quote(script)} {shlex.quote(directory)} "
        f"{shlex.quote(json.dumps(assigned))}"
    )
