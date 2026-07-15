from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from filelock import FileLock

from fugue.bench.library import validate_id

CONTEXT_SYSTEMS_DIR = Path("configs") / "fugue" / "context-systems"
DEFAULT_CACHE_ROOT = Path(".fugue") / "cache" / "context" / "v2"
CONTEXT_MANIFEST = "context-manifest.json"
CONTEXT_INDEX = "index.json"

ContextCapability = Literal[
    "prepare", "retrieve", "bind", "ingest", "sequence", "serve"
]
ContextDelivery = Literal["portable", "native_mcp"]
SupportLevel = Literal["supported", "experimental", "not_applicable", "disabled"]
PreflightPhase = Literal["all", "host", "runtime"]


@dataclass(frozen=True)
class ContextSystemSpec:
    id: str
    title: str
    provider: str
    version: str
    capabilities: frozenset[ContextCapability]
    deliveries: frozenset[ContextDelivery]
    serve_deliveries: frozenset[ContextDelivery] = frozenset()
    support: SupportLevel = "supported"
    description: str = ""
    required_env: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ()
    required_packages: tuple[str, ...] = ()
    runtime_image: str | None = None
    license: str | None = None
    license_url: str | None = None
    source_url: str | None = None
    enabled_by_default: bool = True
    requires_license_approval: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = sorted(self.capabilities)
        data["deliveries"] = sorted(self.deliveries)
        data["serve_deliveries"] = sorted(self.serve_deliveries)
        data["path"] = self.path.as_posix() if self.path else None
        return data


@dataclass(frozen=True)
class RepositorySnapshot:
    task_id: str
    repo: str
    commit: str
    checkout: Path
    dataset_id: str = ""


@dataclass(frozen=True)
class PreparedContext:
    system_id: str
    cache_key: str
    path: Path
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    cache_hit: bool = False


@dataclass(frozen=True)
class ContextBinding:
    extra_instruction_paths: tuple[Path, ...] = ()
    mcp_servers: tuple[dict[str, Any], ...] = ()
    delivery: ContextDelivery = "portable"
    managed_runtime: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    mounts: tuple[dict[str, Any], ...] = ()
    compose_files: tuple[Path, ...] = ()
    artifacts: tuple[Any, ...] = ()


def context_behavior_definition(spec: ContextSystemSpec) -> dict[str, Any]:
    """Return only context definition fields that can affect candidate behavior."""

    return {
        "id": spec.id,
        "provider": spec.provider,
        "version": spec.version,
        "config": spec.config,
    }


def context_behavior_digest(spec: ContextSystemSpec) -> str:
    payload = json.dumps(
        context_behavior_definition(spec),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    text: str
    top_k: int = 10
    expected_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalHit:
    path: str
    start_line: int | None = None
    end_line: int | None = None
    score: float | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextEvent:
    sequence_id: str
    episode: int
    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextCheck:
    name: str
    ok: bool
    detail: str
    severity: Literal["required", "warning"] = "required"
    phase: Literal["host", "runtime"] = "host"


@dataclass(frozen=True)
class ContextRuntime:
    repo_root: Path
    cache_root: Path
    env: dict[str, str]
    output_dir: Path | None = None


@dataclass(frozen=True)
class TrialContext:
    experiment_id: str
    workload_id: str
    task_id: str
    harness: str
    run_key: str | None = None


class ContextProvider(Protocol):
    async def preflight(
        self, spec: ContextSystemSpec, runtime: ContextRuntime
    ) -> list[ContextCheck]: ...

    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext: ...

    async def bind(
        self,
        spec: ContextSystemSpec,
        prepared: PreparedContext,
        trial: TrialContext,
        runtime: ContextRuntime,
        delivery: ContextDelivery,
    ) -> ContextBinding: ...

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]: ...

    async def ingest(
        self,
        spec: ContextSystemSpec,
        event: ContextEvent,
        namespace: Path,
        runtime: ContextRuntime,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class BaseContextProvider:
    async def preflight(
        self, spec: ContextSystemSpec, runtime: ContextRuntime
    ) -> list[ContextCheck]:
        checks = [_command_check(command) for command in spec.required_commands]
        checks.extend(_package_check(package) for package in spec.required_packages)
        checks.extend(
            ContextCheck(
                name=f"env:{name}",
                ok=bool(runtime.env.get(name, "").strip()),
                detail=f"{name} is {'present' if runtime.env.get(name, '').strip() else 'missing'}",
            )
            for name in spec.required_env
        )
        approved = _license_approved(spec, runtime.env)
        if spec.requires_license_approval:
            checks.append(
                ContextCheck(
                    name="license",
                    ok=approved,
                    detail=(
                        f"approved by FUGUE_LICENSE_APPROVED_{_env_id(spec.id)}"
                        if approved
                        else f"{spec.license or 'restricted'} license requires explicit approval"
                    ),
                )
            )
        return checks

    async def bind(
        self,
        spec: ContextSystemSpec,
        prepared: PreparedContext,
        trial: TrialContext,
        runtime: ContextRuntime,
        delivery: ContextDelivery,
    ) -> ContextBinding:
        binding = spec.config.get("binding") or {}
        instruction_paths = tuple(
            _resolve_template_path(path, prepared, runtime)
            for path in _list(binding.get("extra_instruction_paths"))
        )
        declared_mcp_servers = tuple(
            _expand_value(item, spec=spec, prepared=prepared, runtime=runtime)
            for item in _dict_list(binding.get("mcp_servers"))
        )
        managed_runtime = str(binding.get("managed_runtime") or "") or None
        if managed_runtime not in {None, "fugue_context"}:
            raise ValueError(
                f"context system {spec.id} declares unknown managed runtime "
                f"{managed_runtime}"
            )
        if delivery == "portable" and declared_mcp_servers and not managed_runtime:
            raise ValueError(
                f"context system {spec.id} has no portable implementation"
            )
        mcp_servers = declared_mcp_servers if delivery == "native_mcp" else ()
        env = {
            str(key): str(
                _expand_value(value, spec=spec, prepared=prepared, runtime=runtime)
            )
            for key, value in _dict(binding.get("env")).items()
        }
        return ContextBinding(
            extra_instruction_paths=instruction_paths,
            mcp_servers=mcp_servers,
            delivery=delivery,
            managed_runtime=managed_runtime,
            env=env,
            mounts=tuple(_dict_list(binding.get("mounts"))),
            compose_files=tuple(Path(item) for item in _list(binding.get("compose_files"))),
            artifacts=tuple(_list(binding.get("artifacts"))),
        )

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]:
        raise NotImplementedError(f"{spec.id} does not expose ranked retrieval")

    async def ingest(
        self,
        spec: ContextSystemSpec,
        event: ContextEvent,
        namespace: Path,
        runtime: ContextRuntime,
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{spec.id} does not support ingestion")

    async def close(self) -> None:
        return None


class EmptyContextProvider(BaseContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        output.mkdir(parents=True, exist_ok=True)
        return _prepared(spec, snapshot, runtime, {"files": 0, "bytes": 0})

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]:
        return []

    async def ingest(
        self,
        spec: ContextSystemSpec,
        event: ContextEvent,
        namespace: Path,
        runtime: ContextRuntime,
    ) -> dict[str, Any]:
        return {"write_latency_ms": 0.0, "storage_bytes": 0, "context_items": 0}


class AgentsMdContextProvider(BaseContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        artifact = output / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        entries = [
            path.name + ("/" if path.is_dir() else "")
            for path in sorted(snapshot.checkout.iterdir(), key=lambda item: item.name.lower())
            if path.name not in _IGNORED_DIRS
        ][:60]
        interesting = [
            path.relative_to(snapshot.checkout).as_posix()
            for path in snapshot.checkout.rglob("*")
            if path.is_file()
            and path.name in _INTERESTING_FILES
            and not any(part in _IGNORED_DIRS for part in path.parts)
        ][:80]
        body = "\n".join(
            [
                "# Repository Context",
                "",
                f"- Repository: `{snapshot.repo}`",
                f"- Commit: `{snapshot.commit}`",
                "",
                "## Top-level map",
                "",
                *[f"- `{item}`" for item in entries],
                "",
                "## Configuration and entry points",
                "",
                *[f"- `{item}`" for item in interesting],
                "",
            ]
        )
        (artifact / "AGENTS.md").write_text(body)
        return _prepared(
            spec,
            snapshot,
            runtime,
            {"files": 1, "entries": len(entries), "bytes": len(body.encode())},
        )


class MarkdownLogContextProvider(BaseContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        artifact = output / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "MEMORY.md").write_text("# Experiment Memory\n\n")
        return _prepared(spec, snapshot, runtime, {"files": 1, "bytes": 21})

    async def ingest(
        self,
        spec: ContextSystemSpec,
        event: ContextEvent,
        namespace: Path,
        runtime: ContextRuntime,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        namespace.mkdir(parents=True, exist_ok=True)
        path = namespace / "MEMORY.md"
        with path.open("a") as handle:
            handle.write(
                f"\n## Episode {event.episode}: {event.kind}\n\n{event.content.strip()}\n"
            )
        return {
            "write_latency_ms": (time.perf_counter() - started) * 1000,
            "storage_bytes": path.stat().st_size,
            "context_items": _count_headings(path),
        }

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]:
        path = prepared.path / "MEMORY.md"
        if not path.is_file():
            path = prepared.path / "artifact" / "MEMORY.md"
        if not path.is_file():
            return []
        sections = re.split(r"(?=^## )", path.read_text(), flags=re.MULTILINE)
        chunks = [
            {
                "id": f"MEMORY.md:{index}",
                "path": "MEMORY.md",
                "start_line": None,
                "end_line": None,
                "text": section,
            }
            for index, section in enumerate(sections)
            if section.strip()
        ]
        return [
            RetrievalHit(
                path=item["path"],
                score=float(item["score"]),
                text=item["text"],
                metadata={"chunk_id": item["id"]},
            )
            for item in _bm25(query.text, chunks)[: query.top_k]
        ]


class LatMdContextProvider(MarkdownLogContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        artifact = output / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            subprocess.run,
            ["npx", "-y", "lat.md@0.11.0", "init"],
            cwd=artifact,
            env=runtime.env,
            check=True,
            capture_output=True,
            text=True,
        )
        lattice = artifact / "lat.md"
        lattice.mkdir(parents=True, exist_ok=True)
        episodes = lattice / "episodes.md"
        if not episodes.exists():
            episodes.write_text("# Experiment Context\n\n")
        return _prepared(spec, snapshot, runtime, _tree_metrics(artifact))

    async def ingest(
        self,
        spec: ContextSystemSpec,
        event: ContextEvent,
        namespace: Path,
        runtime: ContextRuntime,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        namespace.mkdir(parents=True, exist_ok=True)
        path = namespace / "lat.md" / "episodes.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(
                f"\n## Episode {event.episode}: {event.kind}\n\n"
                f"{event.content.strip()}\n"
            )
        return {
            "write_latency_ms": (time.perf_counter() - started) * 1000,
            "storage_bytes": path.stat().st_size,
            "context_items": _count_headings(path),
        }

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]:
        result = await asyncio.to_thread(
            subprocess.run,
            ["npx", "-y", "lat.md@0.11.0", "search", query.text],
            cwd=prepared.path,
            env=runtime.env,
            capture_output=True,
            text=True,
            check=True,
        )
        text = result.stdout.strip()
        return [
            RetrievalHit(
                path="lat.md",
                text=text,
                metadata={"retriever": "lat search"},
            )
        ] if text else []


class Mem0ContextProvider(BaseContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        (output / "artifact").mkdir(parents=True, exist_ok=True)
        return _prepared(spec, snapshot, runtime, {"files": 0, "bytes": 0})

    async def ingest(
        self,
        spec: ContextSystemSpec,
        event: ContextEvent,
        namespace: Path,
        runtime: ContextRuntime,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        memory = _mem0_memory(spec, namespace, runtime)
        result = await asyncio.to_thread(
            memory.add,
            event.content,
            user_id="fugue",
            metadata={
                "sequence_id": event.sequence_id,
                "episode": event.episode,
                "kind": event.kind,
                **event.metadata,
            },
        )
        return {
            "write_latency_ms": (time.perf_counter() - started) * 1000,
            "storage_bytes": _tree_metrics(namespace)["bytes"],
            "context_items": len((result or {}).get("results", []))
            if isinstance(result, dict)
            else None,
        }

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]:
        memory = _mem0_memory(spec, prepared.path, runtime)
        result = await asyncio.to_thread(
            memory.search, query.text, user_id="fugue", limit=query.top_k
        )
        values = result.get("results", []) if isinstance(result, dict) else result
        return [
            RetrievalHit(
                path=str((item.get("metadata") or {}).get("path") or "MEMORY.md"),
                score=float(item["score"]) if item.get("score") is not None else None,
                text=str(item.get("memory") or item.get("text") or ""),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in values[: query.top_k]
            if isinstance(item, dict)
        ]


class GraphitiContextProvider(BaseContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        (output / "artifact").mkdir(parents=True, exist_ok=True)
        return _prepared(spec, snapshot, runtime, {"files": 0, "bytes": 0})

    async def ingest(
        self,
        spec: ContextSystemSpec,
        event: ContextEvent,
        namespace: Path,
        runtime: ContextRuntime,
    ) -> dict[str, Any]:
        from datetime import UTC, datetime

        from graphiti_core.nodes import EpisodeType

        graph = _graphiti(spec, runtime)
        started = time.perf_counter()
        try:
            await graph.add_episode(
                name=f"{event.sequence_id}:{event.episode}",
                episode_body=event.content,
                source=EpisodeType.text,
                source_description=event.kind,
                reference_time=datetime.now(UTC),
                group_id=_namespace_id(namespace),
            )
        finally:
            await graph.close()
        return {"write_latency_ms": (time.perf_counter() - started) * 1000}

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]:
        graph = _graphiti(spec, runtime)
        try:
            values = await graph.search(
                query.text,
                group_ids=[_namespace_id(prepared.path)],
                num_results=query.top_k,
            )
        finally:
            await graph.close()
        return [
            RetrievalHit(
                path="GRAPHITI",
                score=float(getattr(item, "score", 0) or 0),
                text=str(getattr(item, "fact", None) or getattr(item, "name", item)),
                metadata={"uuid": str(getattr(item, "uuid", ""))},
            )
            for item in values
        ]


class CommandContextProvider(BaseContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        artifact = output / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        prepare = _dict(spec.config.get("prepare"))
        command = _command(prepare.get("command"))
        if command:
            expanded = [
                _format_token(token, spec, snapshot, artifact, runtime) for token in command
            ]
            result = await asyncio.to_thread(
                subprocess.run,
                expanded,
                cwd=snapshot.checkout,
                env=_command_env(runtime.env, prepare.get("env")),
                check=True,
                capture_output=bool(prepare.get("stdout_path")),
                text=bool(prepare.get("stdout_path")),
            )
            if prepare.get("stdout_path"):
                target = artifact / str(prepare["stdout_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(result.stdout or "")
        for value in _list(prepare.get("copy_paths")):
            source = snapshot.checkout / str(value)
            if not source.exists():
                continue
            target = artifact / source.name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
        metrics = _tree_metrics(artifact)
        metrics["command"] = command
        return _prepared(spec, snapshot, runtime, metrics)

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]:
        retrieve = _dict(spec.config.get("retrieve"))
        command = _command(retrieve.get("command"))
        if not command:
            return []
        expanded = [
            token.format(
                query=query.text,
                top_k=query.top_k,
                prepared=prepared.path.as_posix(),
                artifact=(prepared.path / "artifact").as_posix(),
            )
            for token in command
        ]
        result = await asyncio.to_thread(
            subprocess.run,
            expanded,
            cwd=prepared.path / "artifact",
            env=runtime.env,
            check=True,
            capture_output=True,
            text=True,
        )
        return _parse_hits(result.stdout, retrieve.get("format", "json"), query.top_k)


class GitNexusContextProvider(CommandContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        artifact = output / "artifact"
        repository = artifact / "repository"
        home = artifact / "home"
        repository.mkdir(parents=True, exist_ok=True)
        home.mkdir(parents=True, exist_ok=True)
        env = dict(runtime.env)
        env["HOME"] = home.as_posix()
        await asyncio.to_thread(
            subprocess.run,
            [
                "npx",
                "-y",
                "gitnexus@1.6.3",
                "analyze",
                "--skip-agents-md",
                "--force",
                snapshot.checkout.as_posix(),
            ],
            cwd=snapshot.checkout,
            env=env,
            check=True,
        )
        index = snapshot.checkout / ".gitnexus"
        if not index.is_dir():
            raise FileNotFoundError("GitNexus did not create .gitnexus")
        await asyncio.to_thread(
            shutil.copytree,
            snapshot.checkout,
            repository,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".gitnexus", ".git"),
        )
        shutil.copytree(index, repository / ".gitnexus")
        git_dir = snapshot.checkout / ".git"
        if git_dir.is_dir():
            shutil.copytree(git_dir, repository / ".git")
        registry = home / ".gitnexus" / "registry.json"
        if not registry.is_file():
            raise FileNotFoundError("GitNexus did not register the prepared repository")
        text = registry.read_text().replace(
            snapshot.checkout.resolve().as_posix(),
            "/fugue-context/artifact/repository",
        )
        registry.write_text(text)
        return _prepared(spec, snapshot, runtime, _tree_metrics(artifact))


class RagContextProvider(BaseContextProvider):
    async def prepare(
        self,
        spec: ContextSystemSpec,
        snapshot: RepositorySnapshot,
        runtime: ContextRuntime,
    ) -> PreparedContext:
        output = _require_output_dir(runtime)
        artifact = output / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        chunks = _repository_chunks(
            snapshot.checkout,
            lines_per_chunk=int(spec.config.get("lines_per_chunk") or 80),
            overlap=int(spec.config.get("line_overlap") or 20),
        )
        if not chunks:
            raise ValueError(
                f"{spec.id} found no indexable repository text under "
                f"{snapshot.checkout}"
            )
        chunks_path = artifact / "chunks.jsonl"
        with chunks_path.open("w") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk, sort_keys=True) + "\n")

        mode = str(spec.config.get("mode") or "bm25")
        if mode in {"dense", "hybrid"}:
            await asyncio.to_thread(
                _build_lance_index,
                artifact,
                chunks,
                str(spec.config.get("embedding_model") or "BAAI/bge-small-en-v1.5"),
            )
        metrics = _tree_metrics(artifact)
        metrics.update(
            {
                "chunks": len(chunks),
                "files": len({str(chunk["path"]) for chunk in chunks}),
                "embedding_model": spec.config.get("embedding_model"),
                "mode": mode,
            }
        )
        return _prepared(spec, snapshot, runtime, metrics)

    async def retrieve(
        self,
        spec: ContextSystemSpec,
        query: RetrievalQuery,
        prepared: PreparedContext,
        runtime: ContextRuntime,
    ) -> list[RetrievalHit]:
        artifact = prepared.path / "artifact"
        if not (artifact / "chunks.jsonl").is_file():
            artifact = prepared.path
        chunks = [
            json.loads(line)
            for line in (artifact / "chunks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        mode = str(spec.config.get("mode") or "bm25")
        lexical = _bm25(query.text, chunks)
        if mode == "bm25":
            ranked = lexical[: query.top_k]
        else:
            dense = await asyncio.to_thread(
                _dense_search,
                artifact,
                query.text,
                str(spec.config.get("embedding_model") or "BAAI/bge-small-en-v1.5"),
                max(query.top_k, 20),
            )
            ranked = dense if mode == "dense" else _reciprocal_rank_fusion(lexical, dense)
            ranked = ranked[: query.top_k]
        return [
            RetrievalHit(
                path=item["path"],
                start_line=item.get("start_line"),
                end_line=item.get("end_line"),
                score=float(item.get("score") or 0),
                text=item.get("text"),
                metadata={"chunk_id": item.get("id"), "retriever": mode},
            )
            for item in ranked
        ]

    async def ingest(
        self,
        spec: ContextSystemSpec,
        event: ContextEvent,
        namespace: Path,
        runtime: ContextRuntime,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        namespace.mkdir(parents=True, exist_ok=True)
        path = namespace / "chunks.jsonl"
        chunk = {
            "id": f"episode:{event.episode}",
            "path": "MEMORY.md",
            "start_line": event.episode,
            "end_line": event.episode,
            "text": event.content,
        }
        with path.open("a") as handle:
            handle.write(json.dumps(chunk, sort_keys=True) + "\n")
        mode = str(spec.config.get("mode") or "bm25")
        if mode in {"dense", "hybrid"}:
            chunks = [json.loads(line) for line in path.read_text().splitlines()]
            await asyncio.to_thread(
                _build_lance_index,
                namespace,
                chunks,
                str(spec.config.get("embedding_model") or "BAAI/bge-small-en-v1.5"),
            )
        return {
            "write_latency_ms": (time.perf_counter() - started) * 1000,
            "storage_bytes": _tree_metrics(namespace)["bytes"],
            "context_items": event.episode,
        }


def context_system_root(repo_root: Path | None = None) -> Path:
    requested = (repo_root or Path.cwd()) / CONTEXT_SYSTEMS_DIR
    if requested.exists():
        return requested
    bundled = Path(__file__).resolve().parents[2] / CONTEXT_SYSTEMS_DIR
    return bundled if bundled.exists() else requested


def list_context_systems(repo_root: Path | None = None) -> list[ContextSystemSpec]:
    root = context_system_root(repo_root)
    if not root.exists():
        return []
    return [load_context_system(path) for path in sorted(root.glob("*.yaml"))]


def get_context_system(system_id: str, repo_root: Path | None = None) -> ContextSystemSpec:
    system_id = validate_id(system_id, kind="context system id")
    path = context_system_root(repo_root) / f"{system_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"context system not found: {system_id}")
    return load_context_system(path)


def load_context_system(path: Path) -> ContextSystemSpec:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: context system must be a mapping")
    known = {
        "id", "title", "description", "provider", "version", "capabilities",
        "deliveries", "serve_deliveries", "support", "required_env",
        "required_commands", "required_packages", "runtime_image", "license",
        "license_url", "source_url", "enabled_by_default",
        "requires_license_approval", "config",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown fields: {', '.join(unknown)}")
    system_id = validate_id(raw.get("id") or path.stem, kind="context system id")
    capabilities = frozenset(str(item) for item in _list(raw.get("capabilities")))
    allowed = {"prepare", "retrieve", "bind", "ingest", "sequence", "serve"}
    invalid = sorted(capabilities - allowed)
    if invalid:
        raise ValueError(f"{path}: unknown capabilities: {', '.join(invalid)}")
    deliveries = frozenset(str(item) for item in _list(raw.get("deliveries")))
    invalid_deliveries = sorted(deliveries - {"portable", "native_mcp"})
    if invalid_deliveries or not deliveries:
        detail = ", ".join(invalid_deliveries) or "none declared"
        raise ValueError(f"{path}: invalid context deliveries: {detail}")
    serve_deliveries = frozenset(
        str(item) for item in _list(raw.get("serve_deliveries"))
    )
    if not serve_deliveries <= deliveries:
        raise ValueError(f"{path}: serve_deliveries must be supported deliveries")
    if serve_deliveries and "serve" not in capabilities:
        raise ValueError(f"{path}: serve_deliveries require the serve capability")
    provider = str(raw.get("provider") or "").strip()
    version = str(raw.get("version") or "").strip()
    if not provider or not version:
        raise ValueError(f"{path}: provider and version are required")
    support = str(raw.get("support") or "supported")
    if support not in {"supported", "experimental", "not_applicable", "disabled"}:
        raise ValueError(f"{path}: unknown support level {support!r}")
    return ContextSystemSpec(
        id=system_id,
        title=str(raw.get("title") or system_id),
        description=str(raw.get("description") or ""),
        provider=provider,
        version=version,
        capabilities=capabilities,  # type: ignore[arg-type]
        deliveries=deliveries,  # type: ignore[arg-type]
        serve_deliveries=serve_deliveries,  # type: ignore[arg-type]
        support=support,  # type: ignore[arg-type]
        required_env=tuple(str(item) for item in _list(raw.get("required_env"))),
        required_commands=tuple(
            str(item) for item in _list(raw.get("required_commands"))
        ),
        required_packages=tuple(
            str(item) for item in _list(raw.get("required_packages"))
        ),
        runtime_image=_optional_str(raw.get("runtime_image")),
        license=_optional_str(raw.get("license")),
        license_url=_optional_str(raw.get("license_url")),
        source_url=_optional_str(raw.get("source_url")),
        enabled_by_default=bool(raw.get("enabled_by_default", True)),
        requires_license_approval=bool(raw.get("requires_license_approval", False)),
        config=_dict(raw.get("config")),
        path=path,
    )


def load_provider(spec: ContextSystemSpec) -> ContextProvider:
    module_name, separator, name = spec.provider.partition(":")
    if not separator:
        module_name, _, name = spec.provider.rpartition(".")
    if not module_name or not name:
        raise ValueError(f"invalid context provider import: {spec.provider}")
    provider_type = getattr(importlib.import_module(module_name), name)
    return provider_type()


def context_cache_key(
    spec: ContextSystemSpec,
    snapshot: RepositorySnapshot,
    runtime: ContextRuntime | None = None,
) -> str:
    env = runtime.env if runtime else {}
    payload = {
        "repo": snapshot.repo,
        "commit": snapshot.commit,
        "dataset": snapshot.dataset_id,
        "task": snapshot.task_id,
        "system": spec.id,
        "version": spec.version,
        "config": spec.config,
        "provider": spec.provider,
        "builder_model": env.get("FUGUE_BUILDER_MODEL")
        or spec.config.get("builder_model"),
        "embedding_model": env.get("FUGUE_EMBEDDING_MODEL")
        or spec.config.get("embedding_model"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def preflight_context(
    spec: ContextSystemSpec,
    runtime: ContextRuntime,
    *,
    phase: PreflightPhase = "all",
) -> list[ContextCheck]:
    provider = load_provider(spec)
    try:
        host_checks = await provider.preflight(spec, runtime)
    finally:
        await provider.close()
    runtime_checks = _runtime_checks(spec)
    if phase == "host":
        return [check for check in host_checks if check.phase == "host"]
    if phase == "runtime":
        return runtime_checks
    return [*host_checks, *runtime_checks]


async def prepare_context(
    spec: ContextSystemSpec,
    snapshot: RepositorySnapshot,
    runtime: ContextRuntime,
    *,
    rebuild: bool = False,
) -> PreparedContext:
    key = context_cache_key(spec, snapshot, runtime)
    final_dir = runtime.cache_root / key
    manifest_path = final_dir / CONTEXT_MANIFEST
    if manifest_path.is_file() and not rebuild:
        prepared = _read_prepared(final_dir)
        await asyncio.to_thread(
            _update_context_index, runtime.cache_root, spec.id, snapshot, prepared
        )
        return replace(prepared, cache_hit=True)

    runtime.cache_root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(runtime.cache_root / f".{key}.lock", timeout=120)
    await asyncio.to_thread(lock.acquire)
    try:
        backup_dir = runtime.cache_root / f".{key}.previous"
        _recover_cache_generation(final_dir, backup_dir)
        if manifest_path.is_file() and not rebuild:
            prepared = _read_prepared(final_dir)
            await asyncio.to_thread(
                _update_context_index, runtime.cache_root, spec.id, snapshot, prepared
            )
            return replace(prepared, cache_hit=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=runtime.cache_root))
        provider = load_provider(spec)
        started = time.perf_counter()
        resources = _BuildResourceSampler()
        await asyncio.to_thread(resources.start)
        try:
            prepared = await provider.prepare(
                spec, snapshot, replace(runtime, output_dir=temp_dir)
            )
            resource_metrics = await asyncio.to_thread(resources.stop)
            metrics = dict(prepared.metrics)
            metrics.setdefault("build_latency_ms", (time.perf_counter() - started) * 1000)
            metrics.setdefault("cpu_time_sec", resource_metrics["cpu_time_sec"])
            metrics.setdefault("max_memory_mb", resource_metrics["max_memory_mb"])
            metrics.setdefault("index_size_bytes", _tree_metrics(temp_dir)["bytes"])
            metrics.setdefault("builder_input_tokens", None)
            metrics.setdefault("builder_output_tokens", None)
            metrics.setdefault("builder_cost_usd", None)
            manifest = {
                **prepared.manifest,
                "schema_version": 2,
                "system": spec.to_dict(),
                "snapshot": {
                    "task_id": snapshot.task_id,
                    "dataset_id": snapshot.dataset_id,
                    "repo": snapshot.repo,
                    "commit": snapshot.commit,
                },
                "routes": {
                    "builder_model": runtime.env.get("FUGUE_BUILDER_MODEL")
                    or spec.config.get("builder_model"),
                    "embedding_model": runtime.env.get("FUGUE_EMBEDDING_MODEL")
                    or spec.config.get("embedding_model"),
                },
                "cache_key": key,
                "metrics": metrics,
            }
            (temp_dir / CONTEXT_MANIFEST).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            _publish_cache_generation(temp_dir, final_dir, backup_dir)
        except Exception:
            await asyncio.to_thread(resources.stop)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            await provider.close()
    finally:
        await asyncio.to_thread(lock.release)

    prepared = _read_prepared(final_dir)
    await asyncio.to_thread(
        _update_context_index, runtime.cache_root, spec.id, snapshot, prepared
    )
    return prepared


async def bind_context(
    spec: ContextSystemSpec,
    prepared: PreparedContext,
    trial: TrialContext,
    runtime: ContextRuntime,
    *,
    delivery: ContextDelivery,
) -> ContextBinding:
    if delivery not in spec.deliveries:
        raise ValueError(
            f"context system {spec.id} does not support {delivery} delivery"
        )
    provider = load_provider(spec)
    try:
        return await provider.bind(spec, prepared, trial, runtime, delivery)
    finally:
        await provider.close()


async def query_context(
    spec: ContextSystemSpec,
    query: RetrievalQuery,
    prepared: PreparedContext,
    runtime: ContextRuntime,
) -> tuple[list[RetrievalHit], dict[str, Any]]:
    provider = load_provider(spec)
    started = time.perf_counter()
    try:
        hits = await provider.retrieve(spec, query, prepared, runtime)
    finally:
        await provider.close()
    return hits, {
        "query_latency_ms": (time.perf_counter() - started) * 1000,
        "result_count": len(hits),
        "result_tokens": sum(_token_count(hit.text or "") for hit in hits),
    }


def run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Operator callers may already own an event loop. Run the small provider
    # lifecycle in a worker thread instead of nesting event loops.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def prepared_from_index(
    cache_root: Path, system_id: str, task_id: str
) -> PreparedContext | None:
    index_path = cache_root / CONTEXT_INDEX
    if not index_path.is_file():
        return None
    raw = json.loads(index_path.read_text())
    key = ((raw.get("latest") or {}).get(system_id) or {}).get(task_id)
    if not key:
        return None
    path = cache_root / key
    return _read_prepared(path) if (path / CONTEXT_MANIFEST).is_file() else None


def expected_prepared_context(
    spec: ContextSystemSpec,
    snapshot: RepositorySnapshot,
    runtime: ContextRuntime,
) -> PreparedContext:
    key = context_cache_key(spec, snapshot, runtime)
    return PreparedContext(
        system_id=spec.id,
        cache_key=key,
        path=runtime.cache_root / key,
        manifest={},
        metrics={},
        cache_hit=(runtime.cache_root / key / CONTEXT_MANIFEST).is_file(),
    )


def checkout_repository(
    *,
    task_id: str,
    repo: str,
    commit: str,
    checkout_root: Path,
    dataset_id: str = "",
    rebuild: bool = False,
) -> RepositorySnapshot:
    identity = hashlib.sha256(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "repo": repo,
                "commit": commit,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:24]
    checkout_root.mkdir(parents=True, exist_ok=True)
    checkout = checkout_root / identity
    marker = checkout / ".fugue-snapshot.json"
    lock = FileLock(checkout_root / f".{identity}.lock", timeout=120)
    with lock:
        if checkout.is_dir() and marker.is_file() and not rebuild:
            data = json.loads(marker.read_text())
            if data.get("repo") == repo and data.get("commit") == commit:
                return RepositorySnapshot(
                    task_id, repo, commit, checkout, dataset_id
                )
        temp = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=checkout_root))
        try:
            url = (
                repo
                if "://" in repo or repo.endswith(".git")
                else f"https://github.com/{repo}.git"
            )
            subprocess.run(
                ["git", "clone", "--no-tags", "--filter=blob:none", url, temp.as_posix()],
                check=True,
            )
            subprocess.run(["git", "checkout", "--detach", commit], cwd=temp, check=True)
            subprocess.run(["git", "remote", "remove", "origin"], cwd=temp, check=False)
            (temp / ".fugue-snapshot.json").write_text(
                json.dumps(
                    {
                        "dataset_id": dataset_id,
                        "task_id": task_id,
                        "repo": repo,
                        "commit": commit,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if checkout.exists():
                shutil.rmtree(checkout)
            os.replace(temp, checkout)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
    return RepositorySnapshot(task_id, repo, commit, checkout, dataset_id)


def _prepared(
    spec: ContextSystemSpec,
    snapshot: RepositorySnapshot,
    runtime: ContextRuntime,
    metrics: dict[str, Any],
) -> PreparedContext:
    output = _require_output_dir(runtime)
    return PreparedContext(
        system_id=spec.id,
        cache_key=context_cache_key(spec, snapshot, runtime),
        path=output,
        manifest={},
        metrics=metrics,
    )


class _BuildResourceSampler:
    def __init__(self, interval_sec: float = 0.05):
        self.interval_sec = interval_sec
        self.root_pid = os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = False
        self._baseline_cpu: float | None = None
        self._latest_cpu: float | None = None
        self._max_rss_kb: int | None = None

    def start(self) -> None:
        self._available = self._sample()
        if not self._available:
            return
        self._baseline_cpu = self._latest_cpu
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.interval_sec * 4))
            self._thread = None
        if self._available:
            self._sample()
        cpu_time = None
        if self._baseline_cpu is not None and self._latest_cpu is not None:
            cpu_time = max(0.0, self._latest_cpu - self._baseline_cpu)
        return {
            "cpu_time_sec": cpu_time,
            "max_memory_mb": (
                self._max_rss_kb / 1024 if self._max_rss_kb is not None else None
            ),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._sample()

    def _sample(self) -> bool:
        snapshot = _process_tree_resources(self.root_pid)
        if snapshot is None:
            return False
        cpu_time, rss_kb = snapshot
        self._latest_cpu = cpu_time
        self._max_rss_kb = max(self._max_rss_kb or 0, rss_kb)
        return True


def _process_tree_resources(root_pid: int) -> tuple[float, int] | None:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss=,time="],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    processes: dict[int, tuple[int, int, float]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            pid, parent, rss_kb = map(int, parts[:3])
            cpu_time = _parse_process_time(parts[3])
        except ValueError:
            continue
        processes[pid] = (parent, rss_kb, cpu_time)
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _, _) in processes.items():
            if pid not in selected and parent in selected:
                selected.add(pid)
                changed = True
    values = [processes[pid] for pid in selected if pid in processes]
    if not values:
        return None
    return sum(value[2] for value in values), sum(value[1] for value in values)


def _parse_process_time(value: str) -> float:
    day_parts = value.split("-", 1)
    days = int(day_parts[0]) if len(day_parts) == 2 else 0
    fields = day_parts[-1].split(":")
    seconds = float(fields[-1])
    minutes = int(fields[-2]) if len(fields) >= 2 else 0
    hours = int(fields[-3]) if len(fields) >= 3 else 0
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _read_prepared(path: Path) -> PreparedContext:
    manifest = json.loads((path / CONTEXT_MANIFEST).read_text())
    return PreparedContext(
        system_id=str((manifest.get("system") or {}).get("id")),
        cache_key=str(manifest["cache_key"]),
        path=path,
        manifest=manifest,
        metrics=dict(manifest.get("metrics") or {}),
    )


def _recover_cache_generation(final_dir: Path, backup_dir: Path) -> None:
    if not backup_dir.exists():
        return
    if final_dir.exists():
        shutil.rmtree(backup_dir)
    else:
        os.replace(backup_dir, final_dir)


def _publish_cache_generation(
    temp_dir: Path, final_dir: Path, backup_dir: Path
) -> None:
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    had_previous = final_dir.exists()
    if had_previous:
        os.replace(final_dir, backup_dir)
    try:
        os.replace(temp_dir, final_dir)
    except Exception:
        if had_previous and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _update_context_index(
    cache_root: Path,
    system_id: str,
    snapshot: RepositorySnapshot,
    prepared: PreparedContext,
) -> None:
    path = cache_root / CONTEXT_INDEX
    cache_root.mkdir(parents=True, exist_ok=True)
    with FileLock(cache_root / ".index.lock", timeout=120):
        raw = json.loads(path.read_text()) if path.is_file() else {"schema_version": 2}
        identity = hashlib.sha256(
            json.dumps(
                {
                    "dataset_id": snapshot.dataset_id,
                    "task_id": snapshot.task_id,
                    "repo": snapshot.repo,
                    "commit": snapshot.commit,
                    "system_id": system_id,
                    "cache_key": prepared.cache_key,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        raw.setdefault("entries", {})[identity] = {
            "dataset_id": snapshot.dataset_id,
            "task_id": snapshot.task_id,
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "system_id": system_id,
            "cache_key": prepared.cache_key,
            "provider": (prepared.manifest.get("system") or {}).get("provider"),
            "version": (prepared.manifest.get("system") or {}).get("version"),
            "config": (prepared.manifest.get("system") or {}).get("config"),
            "builder_model": (prepared.manifest.get("routes") or {}).get(
                "builder_model"
            ),
            "embedding_model": (prepared.manifest.get("routes") or {}).get(
                "embedding_model"
            ),
        }
        raw.setdefault("latest", {}).setdefault(system_id, {})[
            snapshot.task_id
        ] = prepared.cache_key
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)


def _license_approved(spec: ContextSystemSpec, env: dict[str, str]) -> bool:
    if not spec.requires_license_approval:
        return True
    value = env.get(f"FUGUE_LICENSE_APPROVED_{_env_id(spec.id)}", "")
    return value.strip().lower() in {"1", "true", "yes"}


def _env_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", value).upper()


def _require_output_dir(runtime: ContextRuntime) -> Path:
    if runtime.output_dir is None:
        raise ValueError("context runtime has no output directory")
    return runtime.output_dir


def _command_check(command: str) -> ContextCheck:
    executable = shutil.which(command)
    if not executable:
        return ContextCheck(
            name=f"command:{command}",
            ok=False,
            detail=f"{command} is missing",
            phase="host",
        )
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        version = (result.stdout or result.stderr).strip().splitlines()[0][:200]
    except (OSError, subprocess.SubprocessError, IndexError):
        version = "version unavailable"
    return ContextCheck(
        name=f"command:{command}",
        ok=True,
        detail=f"{command} is available ({version})",
        phase="host",
    )


def _package_check(package: str) -> ContextCheck:
    present = importlib.util.find_spec(package) is not None
    return ContextCheck(
        name=f"package:{package}",
        ok=present,
        detail=(
            f"Python package {package} is installed"
            if present
            else f"install the context extra providing {package}"
        ),
        phase="host",
    )


def _runtime_checks(spec: ContextSystemSpec) -> list[ContextCheck]:
    servers = _dict_list((spec.config.get("binding") or {}).get("mcp_servers"))
    if not servers:
        return []
    commands = [
        [str(server.get("command") or ""), *map(str, _list(server.get("args")))]
        for server in servers
    ]
    if any("fugue.context_server" in command for command in commands):
        return [
            ContextCheck(
                name="runtime:fugue-context",
                ok=True,
                detail="provided by the pinned Fugue context runtime image",
                phase="runtime",
            )
        ]
    image = (spec.runtime_image or "").strip()
    pinned = bool(image and ":" in image and not image.endswith(":latest"))
    return [
        ContextCheck(
            name="runtime:image",
            ok=pinned,
            detail=(
                f"provider MCP runtime is pinned to {image}"
                if pinned
                else "provider MCP binding has no pinned runtime_image"
            ),
            phase="runtime",
        )
    ]


def _mem0_memory(
    spec: ContextSystemSpec, namespace: Path, runtime: ContextRuntime
) -> Any:
    from mem0 import Memory

    from fugue.model_plane import (
        bridge_master_key,
    )

    namespace.mkdir(parents=True, exist_ok=True)
    bridge_base = runtime.env.get("FUGUE_BRIDGE_BASE_URL", "").rstrip("/")
    if not bridge_base:
        from fugue.model_plane import BRIDGE_BASE_URL_HOST

        bridge_base = BRIDGE_BASE_URL_HOST
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "fugue-builder",
                "api_key": bridge_master_key(runtime.env),
                "openai_base_url": f"{bridge_base}/v1",
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": spec.config.get(
                    "embedding_model", "BAAI/bge-small-en-v1.5"
                )
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": (namespace / "qdrant").as_posix(),
                "collection_name": "fugue",
                "embedding_model_dims": 384,
            },
        },
    }
    return Memory.from_config(config)


def _graphiti(spec: ContextSystemSpec, runtime: ContextRuntime) -> Any:
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import (
        OpenAIRerankerClient,
    )
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    from fugue.model_plane import (
        bridge_master_key,
    )

    uri = runtime.env.get("FUGUE_GRAPHITI_URI")
    if not uri:
        raise ValueError("FUGUE_GRAPHITI_URI is required for the local Graphiti store")
    bridge_base = runtime.env.get("FUGUE_BRIDGE_BASE_URL", "").rstrip("/")
    if not bridge_base:
        from fugue.model_plane import BRIDGE_BASE_URL_HOST

        bridge_base = BRIDGE_BASE_URL_HOST
    llm_config = LLMConfig(
        api_key=bridge_master_key(runtime.env),
        model="fugue-builder",
        small_model="fugue-builder",
        base_url=f"{bridge_base}/v1",
        temperature=0,
    )
    llm_client = OpenAIGenericClient(config=llm_config)

    class LocalEmbedder(EmbedderClient):
        def __init__(self) -> None:
            from fastembed import TextEmbedding

            self.model = TextEmbedding(
                model_name=str(
                    spec.config.get("embedding_model")
                    or "BAAI/bge-small-en-v1.5"
                )
            )

        async def create(self, input_data: Any) -> list[float]:
            values = [str(input_data)] if isinstance(input_data, str) else list(input_data)
            vectors = await asyncio.to_thread(lambda: list(self.model.embed(values)))
            return vectors[0].tolist()

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            vectors = await asyncio.to_thread(
                lambda: list(self.model.embed(input_data_list))
            )
            return [vector.tolist() for vector in vectors]

    return Graphiti(
        uri,
        runtime.env.get("FUGUE_GRAPHITI_USER", "neo4j"),
        runtime.env.get("FUGUE_GRAPHITI_PASSWORD", "fugue-local"),
        llm_client=llm_client,
        embedder=LocalEmbedder(),
        cross_encoder=OpenAIRerankerClient(
            client=llm_client,
            config=llm_config,
        ),
    )


def _namespace_id(path: Path) -> str:
    return hashlib.sha256(path.as_posix().encode()).hexdigest()[:24]


def _command(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("context commands must be string arrays")
    return list(value)


def _format_token(
    token: str,
    spec: ContextSystemSpec,
    snapshot: RepositorySnapshot,
    artifact: Path,
    runtime: ContextRuntime,
) -> str:
    return token.format(
        repo=snapshot.checkout.as_posix(),
        output=artifact.as_posix(),
        task=snapshot.task_id,
        commit=snapshot.commit,
        system=spec.id,
        cache_root=runtime.cache_root.as_posix(),
    )


def _command_env(base: dict[str, str], values: Any) -> dict[str, str]:
    env = dict(base)
    for key, value in _dict(values).items():
        text = str(value)
        if text.startswith("${") and text.endswith("}"):
            source = text[2:-1]
            if source == "LITELLM_MASTER_KEY":
                # The bridge has a deliberate local default. Command adapters
                # must resolve the same value instead of turning an absent
                # process variable into an empty credential.
                from fugue.model_plane import bridge_master_key

                text = bridge_master_key(base)
            else:
                text = base.get(source, "")
        env[str(key)] = text
    return env


def _resolve_template_path(
    value: Any, prepared: PreparedContext, runtime: ContextRuntime
) -> Path:
    path = Path(
        str(value).format(
            prepared=prepared.path.as_posix(),
            artifact=(prepared.path / "artifact").as_posix(),
            cache_root=runtime.cache_root.as_posix(),
        )
    )
    return path if path.is_absolute() else runtime.repo_root / path


def _expand_value(
    value: Any,
    *,
    spec: ContextSystemSpec,
    prepared: PreparedContext,
    runtime: ContextRuntime,
) -> Any:
    if isinstance(value, str):
        return value.format(
            system=spec.id,
            version=spec.version,
            prepared=prepared.path.as_posix(),
            artifact=(prepared.path / "artifact").as_posix(),
            cache_root=runtime.cache_root.as_posix(),
        )
    if isinstance(value, list):
        return [
            _expand_value(item, spec=spec, prepared=prepared, runtime=runtime)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): _expand_value(item, spec=spec, prepared=prepared, runtime=runtime)
            for key, item in value.items()
        }
    return value


def _parse_hits(text: str, format_name: str, top_k: int) -> list[RetrievalHit]:
    if not text.strip():
        return []
    if format_name == "jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        values = parsed if isinstance(parsed, list) else parsed.get("hits", [])
    return [
        RetrievalHit(
            path=str(item.get("path") or item.get("file") or ""),
            start_line=_optional_int(item.get("start_line")),
            end_line=_optional_int(item.get("end_line")),
            score=float(item["score"]) if item.get("score") is not None else None,
            text=_optional_str(item.get("text") or item.get("content")),
            metadata=_dict(item.get("metadata")),
        )
        for item in values[:top_k]
        if isinstance(item, dict) and (item.get("path") or item.get("file"))
    ]


_IGNORED_DIRS = {
    ".git",
    ".fugue",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
}
_INTERESTING_FILES = {
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "tox.ini",
    "pytest.ini",
}
_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def _repository_chunks(
    root: Path, *, lines_per_chunk: int, overlap: int
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    step = max(1, lines_per_chunk - overlap)
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root)
        if not path.is_file() or any(
            part in _IGNORED_DIRS for part in relative_path.parts
        ):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES or path.stat().st_size > 1_000_000:
            continue
        try:
            lines = path.read_text(errors="strict").splitlines()
        except (OSError, UnicodeError):
            continue
        relative = relative_path.as_posix()
        for start in range(0, len(lines) or 1, step):
            selected = lines[start : start + lines_per_chunk]
            if not selected:
                break
            chunks.append(
                {
                    "id": f"{relative}:{start + 1}",
                    "path": relative,
                    "start_line": start + 1,
                    "end_line": start + len(selected),
                    "text": "\n".join(selected),
                }
            )
            if start + lines_per_chunk >= len(lines):
                break
    return chunks


def _bm25(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = _tokens(query)
    if not tokens or not chunks:
        return []
    documents = [_tokens(str(chunk.get("text") or "")) for chunk in chunks]
    document_frequency = Counter(
        token for document in documents for token in set(document)
    )
    avg_length = sum(len(document) for document in documents) / len(documents)
    scored: list[dict[str, Any]] = []
    for chunk, document in zip(chunks, documents, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for token in tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document) / max(avg_length, 1))
            score += idf * (frequency * 2.5 / denominator)
        if score:
            scored.append({**chunk, "score": score})
    return sorted(scored, key=lambda item: (-float(item["score"]), item["id"]))


def _build_lance_index(
    artifact: Path, chunks: list[dict[str, Any]], embedding_model: str
) -> None:
    import lancedb
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=embedding_model)
    vectors = list(embedder.embed([str(chunk["text"]) for chunk in chunks]))
    rows = [
        {**chunk, "vector": vector.tolist() if hasattr(vector, "tolist") else list(vector)}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    database = lancedb.connect((artifact / "lancedb").as_posix())
    database.create_table("chunks", data=rows, mode="overwrite")


def _dense_search(
    artifact: Path, query: str, embedding_model: str, limit: int
) -> list[dict[str, Any]]:
    import lancedb
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=embedding_model)
    vector = next(iter(embedder.embed([query])))
    database = lancedb.connect((artifact / "lancedb").as_posix())
    rows = database.open_table("chunks").search(vector).limit(limit).to_list()
    return [
        {
            **row,
            "score": 1.0 / (1.0 + float(row.get("_distance") or 0)),
        }
        for row in rows
    ]


def _reciprocal_rank_fusion(
    first: list[dict[str, Any]], second: list[dict[str, Any]], k: int = 60
) -> list[dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    scores: Counter[str] = Counter()
    for ranking in (first, second):
        for rank, item in enumerate(ranking, start=1):
            item_id = str(item["id"])
            values[item_id] = item
            scores[item_id] += 1.0 / (k + rank)
    return [
        {**values[item_id], "score": score}
        for item_id, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text.lower())


def _token_count(text: str) -> int:
    return max(0, len(text.split()))


def _tree_metrics(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {"files": len(files), "bytes": sum(item.stat().st_size for item in files)}


def _count_headings(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.startswith("## "))


def _dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping, got {type(value).__name__}")
    return dict(value)


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [_dict(item) for item in _list(value)]


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"expected list, got {type(value).__name__}")
    return list(value)


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)
