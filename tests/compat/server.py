from __future__ import annotations

from typing import Any

from fugue.serve.app import create_app
from fugue.serve.runtime import WorkerRequest


class CompatibilityBackend:
    deployment: dict[str, Any] = {
        "deployment_id": "compatibility-fixture",
        "candidate_id": "candidate-compatibility",
        "resources": {"timeout_sec": 30},
    }

    def readiness(self) -> tuple[bool, tuple[str, ...]]:
        return True, ()

    async def run(self, request: WorkerRequest) -> str:
        return "Fugue compatibility response"


app = create_app(
    backend=CompatibilityBackend(),
    env={
        "FUGUE_SERVE_API_KEY": "fugue-compatibility-key",
        "FUGUE_SERVE_CORS_ORIGINS": "http://localhost:3000",
    },
    heartbeat_sec=0.01,
)
