from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import yaml

from fugue.bench import comparison
from fugue.bench.candidates import stable_digest
from fugue.bench.comparison import (
    _approved_comparison_execution_lock,
    _approved_study_intent,
    _qualification_input_readiness,
    check_comparison,
    comparison_from_dict,
    compile_comparison,
    load_comparison,
    preview_comparison,
)
from fugue.reference_studies import registry
from fugue.reference_studies.registry import (
    WANDB_MCP_REFERENCE_STUDY_ID,
    ReferenceStudyBindingV1,
    infer_legacy_reference_study,
    reference_study_binding_from_dict,
    resolve_reference_study_adapter,
)

EXAMPLE = Path("examples/comparisons/source-use-replay")
REFERENCE_STUDY = {
    "id": "wandb-mcp-release",
    "version": 1,
    "intent": "python-package-release-qualification",
}


class _Adapter:
    @staticmethod
    def qualification_input_readiness(
        spec: object,
        *,
        repo_root: Path,
    ) -> tuple[dict[str, str], list[str]]:
        del spec, repo_root
        return {"adapter_input": "a" * 64}, []


def test_explicit_reference_study_is_strict_lazy_and_preview_bound(
    monkeypatch,
) -> None:
    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        return _Adapter

    monkeypatch.setattr(registry, "import_module", fake_import)
    binding = reference_study_binding_from_dict(REFERENCE_STUDY)
    assert imported == []
    assert resolve_reference_study_adapter(binding) is _Adapter
    assert imported == ["fugue.reference_studies.wandb_mcp_qualification"]

    raw = yaml.safe_load((EXAMPLE / "comparison.yaml").read_text())
    raw["execution"]["reference_study"] = REFERENCE_STUDY
    spec = comparison_from_dict(raw, repo_root=Path.cwd(), source=EXAMPLE)
    monkeypatch.setattr(
        comparison,
        "resolve_reference_study_adapter",
        lambda selected: _Adapter,
    )

    readiness = check_comparison(spec, repo_root=Path.cwd())
    preview = preview_comparison(spec, repo_root=Path.cwd())

    assert readiness.qualification_input_digests == {
        "adapter_input": "a" * 64,
        "reference_study_binding": stable_digest(REFERENCE_STUDY),
    }
    assert preview.comparison["execution"]["reference_study"] == REFERENCE_STUDY
    assert (
        preview.readiness["qualification_input_digests"]
        == readiness.qualification_input_digests
    )
    _, _, public_rows = compile_comparison(spec, repo_root=Path.cwd())
    approved = _approved_comparison_execution_lock(
        preview,
        approval_digest="",
        repo_root=Path.cwd(),
        public_rows=public_rows,
    )
    assert approved["reference_study"] == REFERENCE_STUDY
    assert _approved_study_intent(approved) == REFERENCE_STUDY["intent"]


def test_absent_reference_study_preserves_existing_spec_and_generic_dispatch(
    monkeypatch,
) -> None:
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=Path.cwd())
    serialized = spec.to_dict()
    assert "reference_study" not in serialized["execution"]
    assert comparison_from_dict(
        serialized,
        repo_root=Path.cwd(),
        source=Path.cwd(),
    ).spec_digest == spec.spec_digest

    monkeypatch.setattr(
        comparison,
        "resolve_reference_study_adapter",
        lambda binding: (_ for _ in ()).throw(AssertionError(binding)),
    )
    assert _qualification_input_readiness(spec, repo_root=Path.cwd()) == ({}, [])


def test_legacy_quartet_infers_adapter_without_serializing_new_identity(
    monkeypatch,
) -> None:
    spec = load_comparison(EXAMPLE / "comparison.yaml", repo_root=Path.cwd())
    legacy = replace(
        spec,
        execution=replace(
            spec.execution,
            evidence_lock="legacy-evidence-lock.json",
            source_conformance_receipt="legacy-source-conformance.json",
            release_notes_lock="legacy-release-notes.json",
            mechanism_receipt="legacy-mechanism.json",
        ),
    )
    selected: list[ReferenceStudyBindingV1] = []

    def resolve(binding: ReferenceStudyBindingV1) -> _Adapter:
        selected.append(binding)
        return _Adapter()

    monkeypatch.setattr(comparison, "resolve_reference_study_adapter", resolve)
    digests, blockers = _qualification_input_readiness(
        legacy,
        repo_root=Path.cwd(),
    )

    assert blockers == []
    assert digests == {"adapter_input": "a" * 64}
    assert selected[0].id == WANDB_MCP_REFERENCE_STUDY_ID
    assert "reference_study" not in legacy.to_dict()["execution"]
    assert infer_legacy_reference_study(SimpleNamespace()) is None


def test_partial_or_new_release_fields_do_not_infer_wandb_reference_study() -> None:
    partial = SimpleNamespace(
        evidence_lock="generic-evidence-lock.json",
        source_conformance_receipt=None,
        release_notes_lock=None,
        mechanism_receipt=None,
    )
    complete = SimpleNamespace(
        evidence_lock="generic-evidence-lock.json",
        source_conformance_receipt="generic-conformance.json",
        release_notes_lock="generic-release-notes.json",
        mechanism_receipt="generic-mechanism.json",
    )

    assert infer_legacy_reference_study(partial, schema_version=1) is None
    assert infer_legacy_reference_study(complete, schema_version=2) is None


def test_reference_study_rejects_unknown_fields_and_bindings() -> None:
    for raw in (
        {**REFERENCE_STUDY, "release_contract": "mutable.json"},
        {**REFERENCE_STUDY, "version": 0},
        {**REFERENCE_STUDY, "id": "unknown-reference"},
    ):
        try:
            reference_study_binding_from_dict(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid reference study: {raw}")
