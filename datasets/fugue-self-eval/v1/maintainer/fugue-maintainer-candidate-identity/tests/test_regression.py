from fugue.bench.context import ContextSystemSpec
from fugue.bench.job_config import _candidate_id
from fugue.bench.library import ExperimentSpec, FeatureVariant
from fugue.bench.manifest import HarnessSpec
from fugue.model_plane import resolve_model_route


def test_variant_identity_is_part_of_candidate_identity(tmp_path):
    experiment = ExperimentSpec(id="identity", title="Identity")
    context = ContextSystemSpec(
        id="none",
        title="None",
        provider="fugue.bench.context:EmptyContextProvider",
        version="1",
        capabilities=frozenset(),
    )
    harness = HarnessSpec(name="codex", agent="fugue.agents:FugueCodex")
    route = resolve_model_route("openai/gpt-5", {})

    first = _candidate_id(
        experiment=experiment,
        variant=FeatureVariant(id="first", label="First"),
        harness=harness,
        route=route,
        context_spec=context,
        skill_ids=[],
        agent_config_hash="same",
        repo_root=tmp_path,
    )
    second = _candidate_id(
        experiment=experiment,
        variant=FeatureVariant(id="second", label="Second"),
        harness=harness,
        route=route,
        context_spec=context,
        skill_ids=[],
        agent_config_hash="same",
        repo_root=tmp_path,
    )

    assert first != second
