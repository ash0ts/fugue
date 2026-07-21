from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessCapabilities:
    native_mcp: bool
    isolated_home: bool
    provider_independent_tools: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "native_mcp": self.native_mcp,
            "isolated_home": self.isolated_home,
            "provider_independent_tools": self.provider_independent_tools,
        }


_BUILTIN_CAPABILITIES = {
    "fugue.agents:FugueHermes": HarnessCapabilities(True, True, True),
    "fugue.agents:FugueOpenClaw": HarnessCapabilities(True, True, True),
    "fugue.agents:FugueClaudeCode": HarnessCapabilities(True, True, True),
    "fugue.agents:FugueCodex": HarnessCapabilities(True, True, True),
    "fugue.agents:FugueWBAResponses": HarnessCapabilities(False, True, True),
}


def harness_capabilities(agent_import: str) -> HarnessCapabilities:
    return _BUILTIN_CAPABILITIES.get(
        agent_import,
        HarnessCapabilities(False, False, False),
    )
