from fugue.artifacts import (
    artifact_recoveries,
    artifact_source_paths,
    harbor_artifacts,
)


def test_harbor_convention_owns_nested_log_artifacts() -> None:
    values = [
        "/logs/artifacts/fugue-answer.md",
        "/logs/artifacts/context.jsonl",
        "/root/result.json",
        {"source": "/logs/sidecar.jsonl", "service": "sidecar"},
    ]

    assert harbor_artifacts(values) == [
        "/logs/artifacts",
        "/root/result.json",
        {"source": "/logs/sidecar.jsonl", "service": "sidecar"},
    ]
    assert artifact_source_paths(values) == [
        "/logs/artifacts/fugue-answer.md",
        "/logs/artifacts/context.jsonl",
        "/root/result.json",
        "/logs/sidecar.jsonl",
    ]


def test_artifact_recovery_is_limited_to_exact_log_file_mirrors() -> None:
    recoveries = artifact_recoveries(
        [
            "/logs/artifacts/fugue-answer.md",
            "/logs/artifacts",
            "/root/result.json",
            "logs/artifacts/relative.md",
            "/logs/artifacts/../escape.md",
        ],
        "/worktree",
    )

    assert len(recoveries) == 1
    assert recoveries[0].target == "/logs/artifacts/fugue-answer.md"
    assert recoveries[0].candidates == (
        "/worktree/logs/artifacts/fugue-answer.md",
        "/workspace/logs/artifacts/fugue-answer.md",
        "/root/logs/artifacts/fugue-answer.md",
    )
