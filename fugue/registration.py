from __future__ import annotations

import json
import shlex


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
