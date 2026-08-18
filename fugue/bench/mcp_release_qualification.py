"""Legacy import alias for the W&B MCP reference-study implementation.

The qualification is product-specific reference material, not part of
Fugue's generic comparison core.  Keep the historical module path as an
identity-preserving alias so existing imports and monkeypatches operate on
the exact implementation module rather than on a forwarding copy.
"""

from __future__ import annotations

import importlib
import sys

_implementation = importlib.import_module(
    "fugue.reference_studies.wandb_mcp_qualification_core"
)
sys.modules[__name__] = _implementation
