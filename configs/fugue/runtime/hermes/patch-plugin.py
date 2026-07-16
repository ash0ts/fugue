from __future__ import annotations

import hashlib
import json
from pathlib import Path

root = Path("/opt/fugue-agent-runtime/hermes-otel")
target = root / "tracer.py"
source = target.read_text()
needle = "attrs = dict(attributes or {})"
replacement = "attrs = {**(self.config.resource_attributes or {}), **dict(attributes or {})}"
expected_sha256 = "d01e2ca71330fbca465e73d3bb117377ae8a79191d7edfd5441cc9bb47ea0194"
if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
    raise SystemExit("hermes-otel tracer source digest mismatch")
if source.count(needle) != 2:
    raise SystemExit("hermes-otel tracer patch target mismatch")
target.write_text(source.replace(needle, replacement))
(root / "fugue-patch-lock.json").write_text(
    json.dumps({"tracer.py": hashlib.sha256(source.encode()).hexdigest()}, indent=2)
    + "\n"
)
