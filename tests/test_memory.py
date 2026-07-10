from pathlib import Path

from fugue.bench.manifest import TaskSpec
from fugue.bench.memory import AgentsMdBuilder, write_memory_instruction


def test_agentsmd_builder_writes_golden_files(tmp_path: Path) -> None:
    repo = Path(__file__).parent / "fixtures" / "repo"
    artifact = tmp_path / "agentsmd" / "fixture__task"
    task = TaskSpec(
        id="fixture__task",
        repo="fixture/repo",
        base_commit="abc123",
    )

    AgentsMdBuilder().build(repo, task, artifact)

    agents = (artifact / "AGENTS.md").read_text()
    assert "- Task: `fixture__task`" in agents
    assert "- `README.md`" in agents
    assert "- `pyproject.toml`" in agents
    assert (artifact / ".fugue-memory" / "README.md").is_file()


def test_memory_instruction_points_to_injected_memory(tmp_path: Path) -> None:
    instruction = write_memory_instruction(tmp_path, "agentsmd")

    assert instruction is not None
    assert "Additional repository memory" in instruction.read_text()
