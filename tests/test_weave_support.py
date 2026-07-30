from __future__ import annotations

import sys
from types import SimpleNamespace

from fugue import weave_support


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
