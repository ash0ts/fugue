from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from fugue.agent_tracing import (
    agent_conversation_id,
    agent_conversation_name,
    codex_skill_instruction,
    conversation_id,
    normalize_trace_content,
    openclaw_agent_id,
    openclaw_conversation_id,
    skill_invocation_evidence,
    stable_agent_name,
)
from fugue.registration import (
    context_registration_digest,
    skill_registration_probe_command,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_MODEL_PLANE = REPO_ROOT / "fugue" / "agents" / "model_plane.py"


def test_harness_agents_are_stable_and_trials_are_deterministic() -> None:
    assert stable_agent_name("hermes") == "hermes-agent"
    assert stable_agent_name("openclaw") == "openclaw"
    assert stable_agent_name("claude-code") == "claude-code"
    assert stable_agent_name("codex") == "codex"
    first = conversation_id("run:task:codex:trial-1")
    assert first == conversation_id("run:task:codex:trial-1")
    assert first != conversation_id("run:task:codex:trial-2")
    assert str(uuid.UUID(first)) == first


def test_openclaw_trial_identity_preserves_stable_agent_name() -> None:
    run_key = "run:task:openclaw:trial-1"
    fugue_id = "872b3077-0f62-544d-8e09-aa437b84f029"

    assert openclaw_agent_id(fugue_id) == f"fugue-{fugue_id}"
    assert openclaw_conversation_id(fugue_id) == f"agent:fugue-{fugue_id}:main"
    assert agent_conversation_id("openclaw", run_key) == openclaw_conversation_id(
        conversation_id(run_key)
    )
    assert agent_conversation_id("hermes", run_key) == conversation_id(run_key)
    assert stable_agent_name("openclaw") == "openclaw"


def test_conversation_names_are_human_readable_and_bounded() -> None:
    assert (
        agent_conversation_name(
            run_name="standup-skills-full",
            task_id="paper-anonymizer",
            variant_id="with-pdf-skill",
            trial_index=1,
        )
        == "standup-skills-full · paper-anonymizer · with-pdf-skill · t001"
    )
    assert (
        len(
            agent_conversation_name(
                run_name="x" * 300,
                task_id="task",
                variant_id="baseline",
                trial_index=0,
            )
        )
        == 256
    )


def test_model_plane_uses_the_typed_trace_conversation_hook() -> None:
    source = AGENT_MODEL_PLANE.read_text()

    for harness in ("hermes", "openclaw", "claude-code", "codex"):
        assert f'TRACE_HARNESS = "{harness}"' in source
    assert '"gen_ai.conversation.id": self.trace_conversation_id' in source
    assert '"weave.conversation.name": agent_conversation_name(' in source
    assert '"fugue.conversation_id": self.trace_conversation_id' in source
    assert '"planned_conversation_id": self.trace_conversation_id' in source
    assert '"FUGUE_WEAVE_CONVERSATION_ID": self.conversation_id' not in source
    assert "dict.fromkeys([self.trace_conversation_id, *native_ids])" in source


def test_model_plane_normalizes_only_declared_artifact_transport() -> None:
    source = AGENT_MODEL_PLANE.read_text()

    assert "FUGUE_EXPECTED_ARTIFACT_PATHS" in source
    assert "artifact_recoveries(expected, repo_root)" in source
    assert '"artifact_normalization"' in source


def test_trace_content_is_explicit() -> None:
    assert normalize_trace_content(None) == "full"
    assert normalize_trace_content(" metadata ") == "metadata"
    with pytest.raises(ValueError, match="full.*metadata"):
        normalize_trace_content("redacted")


def test_skill_registration_probe_requires_every_assigned_skill(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    (root / "pdf").mkdir(parents=True)
    (root / "pdf" / "SKILL.md").write_text("# PDF\n")

    complete = subprocess.run(
        skill_registration_probe_command(root.as_posix(), ["pdf"]),
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(complete.stdout)

    assert complete.returncode == 0
    assert payload["skills_registered"] == ["pdf"]
    assert payload["registration_digest"].startswith("sha256:")

    incomplete = subprocess.run(
        skill_registration_probe_command(root.as_posix(), ["pdf", "missing"]),
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    assert incomplete.returncode == 2

    wrong = subprocess.run(
        skill_registration_probe_command(root.as_posix(), ["other"]),
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    wrong_payload = json.loads(wrong.stdout)
    assert wrong.returncode == 2
    assert wrong_payload["missing_skills"] == ["other"]
    assert wrong_payload["unexpected_skills"] == ["pdf"]


def test_skill_registration_probe_resolves_agent_home(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent-home"
    skill = home / ".hermes" / "skills" / "pdf-artifact-workflow"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# PDF\n")

    result = subprocess.run(
        skill_registration_probe_command(
            "$HOME/.hermes/skills", ["pdf-artifact-workflow"]
        ),
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": home.as_posix()},
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["directory"] == (home / ".hermes" / "skills").as_posix()
    assert payload["skills_registered"] == ["pdf-artifact-workflow"]


def test_codex_skill_read_is_normalized_from_a_successful_structured_event(
    tmp_path: Path,
) -> None:
    skill_root = "/tmp/isolated/home/.agents/skills"
    (tmp_path / "codex.txt").write_text(
        "not json\n"
        + json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_6",
                    "type": "command_execution",
                    "command": (
                        "/bin/bash -lc 'cat "
                        f"{skill_root}/pdf-artifact-workflow/SKILL.md'"
                    ),
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        )
        + "\n"
    )

    evidence = skill_invocation_evidence(
        tmp_path,
        "codex",
        {
            "skills_assigned": ["pdf-artifact-workflow"],
            "skills_registered": ["pdf-artifact-workflow"],
            "directory": skill_root,
        },
    )

    assert evidence == {
        "status": "observed",
        "skills_invoked": ["pdf-artifact-workflow"],
        "missing_skills": [],
        "events": [
            {
                "item_id": "item_6",
                "operation": "read_skill_instructions",
                "skill_id": "pdf-artifact-workflow",
            }
        ],
    }


def test_codex_assigned_skills_are_read_before_task_work() -> None:
    instruction = codex_skill_instruction(
        "Fill the PDF.",
        skills=["pdf-artifact-workflow"],
        directory="/tmp/cell/home/.agents/skills",
    )

    assert instruction.index(
        "cat /tmp/cell/home/.agents/skills/pdf-artifact-workflow/SKILL.md"
    ) < instruction.index("Fill the PDF.")
    assert "then follow the instructions you read" in instruction
    assert (
        codex_skill_instruction(
            "Inspect the repository.", skills=[], directory="/tmp/cell/skills"
        )
        == "Inspect the repository."
    )
    with pytest.raises(ValueError, match="invalid skill id"):
        codex_skill_instruction(
            "Unsafe.", skills=["../outside"], directory="/tmp/cell/skills"
        )


def test_codex_skill_evidence_ignores_failed_or_unrelated_commands(
    tmp_path: Path,
) -> None:
    root = "/tmp/isolated/skills"
    events = [
        {
            "type": "item.completed",
            "item": {
                "id": "failed",
                "type": "command_execution",
                "command": f"cat {root}/pdf/SKILL.md",
                "exit_code": 1,
                "status": "failed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "echo",
                "type": "command_execution",
                "command": f"echo {root}/pdf/SKILL.md",
                "exit_code": 0,
                "status": "completed",
            },
        },
    ]
    (tmp_path / "codex.txt").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    evidence = skill_invocation_evidence(
        tmp_path,
        "codex",
        {"skills_assigned": ["pdf"], "directory": root},
    )

    assert evidence["status"] == "not_observed"
    assert evidence["skills_invoked"] == []
    assert evidence["missing_skills"] == ["pdf"]


def test_context_registration_digest_is_order_independent_and_behavioral() -> None:
    inputs = {
        "context_system_id": "gitnexus",
        "delivery": "native_mcp",
        "context_config_hash": "config-a",
        "command": None,
    }
    first = context_registration_digest(
        **inputs,
        servers=[{"name": "b", "url": "http://b"}, {"name": "a"}],
    )
    second = context_registration_digest(
        **inputs,
        servers=[{"name": "a"}, {"name": "b", "url": "http://b"}],
    )
    changed = context_registration_digest(
        **{**inputs, "context_config_hash": "config-b"},
        servers=[{"name": "a"}, {"name": "b", "url": "http://b"}],
    )

    assert first == second
    assert first.startswith("sha256:")
    assert changed != first


def test_trial_trace_attributes_are_flat_and_comparable() -> None:
    source = AGENT_MODEL_PLANE.read_text()

    for attribute in (
        "fugue.run_id",
        "fugue.experiment_id",
        "fugue.workload_id",
        "fugue.harness",
        "fugue.variant_id",
        "fugue.context_system_id",
        "fugue.context_delivery",
        "fugue.context_registration_status",
        "fugue.context_registration_digest",
        "fugue.context_support",
        "fugue.integration_ids",
        "fugue.task_id",
        "fugue.trial_index",
        "fugue.comparison_example_id",
        "fugue.candidate_id",
        "fugue.evaluation_scope_id",
        "fugue.model_provider",
        "fugue.model",
        "fugue.model_wire_protocol",
        "fugue.model_endpoint_kind",
        "fugue.model_upstream_host",
        "fugue.model_bridge_required",
        "fugue.tool_result_modalities",
        "weave.eval.predict_and_score_call_id",
        "weave.eval.project_id",
        "weave.eval.evaluation_name",
    ):
        assert f'"{attribute}"' in source
    assert "FUGUE_TRACE_ATTRIBUTES_JSON" in source
    assert "key: str(value)" in source
    assert "self._resolved_env_vars.update(trace_environment)" in source
    assert 'env.update(self._trace_environment("hermes", self.model_route))' in source
    assert 'env.update(self._trace_environment("openclaw", self.model_route))' in source


def test_hermes_runtime_patch_promotes_resource_attributes_to_spans() -> None:
    patch = (REPO_ROOT / "configs/fugue/runtime/hermes/patch-plugin.py").read_text()

    assert "self.config.resource_attributes or {}" in patch
    assert "hermes-otel tracer patch target mismatch" in patch
    assert "FUGUE_WEAVE_SINGLE_TURN_KEY" in patch
    assert "_finalize_fugue_single_turns" in patch
    assert "hermes-otel turn-end patch target mismatch" in patch


def test_native_plugin_patches_are_pinned_and_integrity_checked() -> None:
    source = AGENT_MODEL_PLANE.read_text()
    codex_runtime = (
        REPO_ROOT / "configs/fugue/runtime/codex/patch-runtime.mjs"
    ).read_text()
    runtime_patches = "\n".join(
        path.read_text()
        for path in (
            REPO_ROOT / "configs/fugue/runtime/openclaw/patch-runtime.mjs",
            REPO_ROOT / "configs/fugue/runtime/claude-code/patch-runtime.mjs",
        )
    )

    assert source.count('_WEAVE_PLUGIN_VERSION = "0.1.1"') == 2
    assert '_WEAVE_PLUGIN_VERSION = "0.2.12"' in source
    assert '_HERMES_VERSION = "v2026.6.5"' in source
    assert '_OPENCLAW_VERSION = "2026.7.1"' in source
    assert '_CLAUDE_CODE_VERSION = "2.1.210"' in source
    assert '_CODEX_VERSION = "0.143.0"' in source
    assert "@latest" not in source
    assert "openclaw plugins install weave-openclaw@" not in source
    assert "hermes-install.sh" not in source
    assert "npm install -g weave-claude-code@" not in source
    assert "OpenClaw prepared runtime is missing" in source
    assert "Hermes prepared runtime is missing" in source
    assert "Claude Code prepared runtime is missing" in source
    assert "Codex prepared runtime is missing" in source
    assert "weave-codex run -- codex exec" in source
    assert "weave-codex install" not in source
    assert 'tool_result_guard_cli_flags(self.model_route, "codex")' in source
    assert "set -o pipefail" in source
    assert "pinned patch target mismatch" in runtime_patches
    assert "key.startsWith('fugue.')" in runtime_patches
    assert "self._resolved_env_vars.update(" in source
    assert "!== 3" in codex_runtime
    assert "codex mcp list --json" in source
    assert '"HOME": f"{remote_codex_home}/home"' in source
    assert 'f"{remote_codex_home}/home/.agents/skills"' in source
    assert "pending_native_registration" in source
    assert '"status": "failed"' in source
    for tool in (
        "web_search",
        "web_fetch",
        "browser",
        "image",
        "memory_search",
        "memory_get",
    ):
        assert f'"{tool}"' in source
    assert "context registration probe failed before agent execution" in source
    assert "trial policy rejected a mounted Docker socket" in source
    assert "_lock_trial_mutators(environment)" in source
    assert '"post_execution"' in source
