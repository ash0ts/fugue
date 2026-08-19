from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from fugue import weave_support


def test_local_evidence_mode_never_initializes_weave_with_a_wandb_key(
    monkeypatch,
) -> None:
    initialized = False
    calls = 0

    def unexpected_initialize(*args, **kwargs):
        nonlocal initialized
        initialized = True
        raise AssertionError(f"local mode initialized Weave: {args}, {kwargs}")

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "local-result"

    monkeypatch.setattr(weave_support, "initialize_weave", unexpected_initialize)

    result = asyncio.run(
        weave_support.trace_async_operation(
            "local-operation",
            {},
            {
                "FUGUE_EVIDENCE_MODE": "local",
                "WANDB_API_KEY": "integration-only-key",
            },
            operation,
            lambda value: {"value": value},
        )
    )

    assert result == "local-result"
    assert calls == 1
    assert initialized is False


def test_initialize_weave_reactivates_destination_after_a_b_a_switch(
    monkeypatch,
) -> None:
    initialized: list[str] = []
    weave = SimpleNamespace(init=initialized.append)
    monkeypatch.setitem(sys.modules, "weave", weave)
    monkeypatch.setattr(weave_support, "_ACTIVE_DESTINATION_DIGEST", None)
    monkeypatch.setattr(
        weave_support,
        "_apply_weave_environment",
        lambda env: None,
    )

    weave_support.initialize_weave("wandb/project-a", {})
    weave_support.initialize_weave("wandb/project-b", {})
    weave_support.initialize_weave("wandb/project-a", {})

    assert initialized == [
        "wandb/project-a",
        "wandb/project-b",
        "wandb/project-a",
    ]


def test_initialize_weave_reactivates_same_project_when_endpoint_changes(
    monkeypatch,
) -> None:
    initialized: list[str] = []
    weave = SimpleNamespace(init=initialized.append)
    monkeypatch.setitem(sys.modules, "weave", weave)
    monkeypatch.setattr(weave_support, "_ACTIVE_DESTINATION_DIGEST", None)
    monkeypatch.setattr(
        weave_support,
        "_apply_weave_environment",
        lambda env: None,
    )

    weave_support.initialize_weave(
        "wandb/project-a",
        {"FUGUE_WEAVE_BASE_URL": "https://api-a.example.test"},
    )
    weave_support.initialize_weave(
        "wandb/project-a",
        {"FUGUE_WEAVE_BASE_URL": "https://api-b.example.test"},
    )

    assert initialized == ["wandb/project-a", "wandb/project-a"]


def test_initialize_weave_does_not_leak_project_identity_to_process_env(
    monkeypatch,
) -> None:
    initialized: list[str] = []
    weave = SimpleNamespace(init=initialized.append)
    monkeypatch.setitem(sys.modules, "weave", weave)
    monkeypatch.setattr(weave_support, "_ACTIVE_DESTINATION_DIGEST", None)
    for key in (
        "FUGUE_WEAVE_PROJECT",
        "WEAVE_PROJECT",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)

    weave_support.initialize_weave("wandb/project-a", {})

    assert initialized == ["wandb/project-a"]
    for key in (
        "FUGUE_WEAVE_PROJECT",
        "WEAVE_PROJECT",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
    ):
        assert key not in weave_support.os.environ


def test_initialize_weave_reactivates_cached_destination_when_client_was_cleared(
    monkeypatch,
) -> None:
    initialized: list[str] = []
    weave = SimpleNamespace(init=initialized.append)
    monkeypatch.setitem(sys.modules, "weave", weave)
    monkeypatch.setattr(weave_support, "_ACTIVE_DESTINATION_DIGEST", None)
    monkeypatch.setattr(
        weave_support,
        "_apply_weave_environment",
        lambda env: None,
    )
    active_projects = iter(
        [
            weave_support._WEAVE_CLIENT_CONTEXT_UNAVAILABLE,
            None,
        ]
    )
    monkeypatch.setattr(
        weave_support,
        "_active_weave_project_slug",
        lambda: next(active_projects),
    )

    weave_support.initialize_weave("wandb/project-a", {})
    weave_support.initialize_weave("wandb/project-a", {})

    assert initialized == ["wandb/project-a", "wandb/project-a"]


def test_weave_destination_session_serializes_and_restores_environment(
    monkeypatch,
) -> None:
    initialized: list[str] = []
    observed: list[tuple[str, str | None]] = []
    first_entered = threading.Event()
    release_first = threading.Event()
    weave = SimpleNamespace(init=initialized.append)
    monkeypatch.setitem(sys.modules, "weave", weave)
    monkeypatch.setattr(weave_support, "_ACTIVE_DESTINATION_DIGEST", None)
    monkeypatch.setattr(
        weave_support,
        "_active_weave_project_slug",
        lambda: weave_support._WEAVE_CLIENT_CONTEXT_UNAVAILABLE,
    )
    monkeypatch.setenv("WANDB_API_KEY", "original-key")

    def publish(project: str, api_key: str, wait: bool) -> None:
        with weave_support.weave_destination_session(
            project,
            {"WANDB_API_KEY": api_key},
        ):
            observed.append((project, weave_support.os.environ.get("WANDB_API_KEY")))
            if wait:
                first_entered.set()
                assert release_first.wait(timeout=5)

    first = threading.Thread(
        target=publish,
        args=("wandb/project-a", "key-a", True),
    )
    second = threading.Thread(
        target=publish,
        args=("wandb/project-b", "key-b", False),
    )
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    assert observed == [("wandb/project-a", "key-a")]
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert observed == [
        ("wandb/project-a", "key-a"),
        ("wandb/project-b", "key-b"),
    ]
    assert weave_support.os.environ["WANDB_API_KEY"] == "original-key"


def test_weave_destination_session_restores_every_owned_key_when_init_fails(
    monkeypatch,
) -> None:
    before = {
        key: f"stale-{index}"
        for index, key in enumerate(weave_support._WEAVE_ENV_KEYS)
    }
    for key, value in before.items():
        monkeypatch.setenv(key, value)
    observed: dict[str, str | None] = {}

    def fail_init(_project: str) -> None:
        observed.update(
            {key: weave_support.os.environ.get(key) for key in before}
        )
        raise RuntimeError("simulated Weave initialization failure")

    monkeypatch.setitem(sys.modules, "weave", SimpleNamespace(init=fail_init))
    monkeypatch.setattr(weave_support, "_ACTIVE_DESTINATION_DIGEST", None)
    monkeypatch.setattr(
        weave_support,
        "_active_weave_project_slug",
        lambda: weave_support._WEAVE_CLIENT_CONTEXT_UNAVAILABLE,
    )

    with pytest.raises(RuntimeError, match="initialization failure"):
        with weave_support.weave_destination_session(
            "wandb/project-a",
            {
                "WANDB_API_KEY": "exact-key",
                "FUGUE_WEAVE_BASE_URL": "https://api-a.example.test",
                "FUGUE_WEAVE_TRACE_SERVER_URL": "https://trace-a.example.test",
                "WANDB_APP_BASE_URL": "https://app-a.example.test",
                "WANDB_INSECURE_DISABLE_SSL": "true",
            },
        ):
            raise AssertionError("failed initialization cannot enter the session")

    assert observed["WANDB_API_KEY"] == "exact-key"
    assert observed["WANDB_BASE_URL"] == "https://api-a.example.test"
    assert observed["WF_TRACE_SERVER_URL"] == "https://trace-a.example.test"
    assert observed["WANDB_APP_BASE_URL"] == "https://app-a.example.test"
    assert observed["WANDB_INSECURE_DISABLE_SSL"] == "true"
    assert observed["WANDB_PUBLIC_BASE_URL"] is None
    assert {
        key: weave_support.os.environ.get(key) for key in before
    } == before


def test_weave_destination_session_rejects_endpoint_drift_before_init(
    monkeypatch,
) -> None:
    initialized: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "weave",
        SimpleNamespace(init=initialized.append),
    )
    monkeypatch.setattr(weave_support, "_ACTIVE_DESTINATION_DIGEST", None)
    monkeypatch.setenv("WANDB_BASE_URL", "https://original.example.test")
    destination = weave_support.resolve_evidence_destination(
        {
            "FUGUE_WEAVE_PROJECT": "wandb/project-a",
            "FUGUE_WEAVE_BASE_URL": "https://api-a.example.test",
            "FUGUE_WEAVE_TRACE_SERVER_URL": "https://trace-a.example.test",
            "WANDB_APP_BASE_URL": "https://app-a.example.test",
        }
    )

    with pytest.raises(ValueError, match="environment disagrees"):
        with weave_support.weave_destination_session(
            destination,
            {
                "FUGUE_WEAVE_BASE_URL": "https://api-b.example.test",
                "FUGUE_WEAVE_TRACE_SERVER_URL": "https://trace-b.example.test",
                "WANDB_APP_BASE_URL": "https://app-b.example.test",
            },
        ):
            raise AssertionError("drifted endpoint cannot enter the session")

    assert initialized == []
    assert (
        weave_support.os.environ["WANDB_BASE_URL"]
        == "https://original.example.test"
    )
