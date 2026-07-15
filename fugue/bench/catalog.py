from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fugue.bench.export import export_rows
from fugue.bench.library import (
    ExperimentSpec,
    get_experiment,
    list_experiments,
)
from fugue.redaction import redact_value

CATALOG_PATH = Path(".fugue/cache/catalog/v2/catalog.sqlite")
FILTER_FIELDS = {
    "record_type",
    "experiment_id",
    "run_id",
    "run_key",
    "workload_id",
    "task_name",
    "harness",
    "variant_id",
    "preset_id",
    "prompt_id",
    "skill_id",
    "integration_id",
    "context_system_id",
    "provider",
    "model",
    "status",
    "intervention_type",
    "source",
}


@dataclass(frozen=True)
class CatalogStatus:
    path: str
    experiments: int
    records: int
    local_records: int
    refreshed_at: str | None
    revision: str | None


@dataclass(frozen=True)
class ArtifactExcerpt:
    path: str
    sha256: str
    text: str
    truncated: bool


class ExperimentCatalog:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.path = self.repo_root / CATALOG_PATH

    def refresh(self) -> CatalogStatus:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._create_schema(connection)
            self._refresh_local(connection)
            now = datetime.now(UTC).isoformat()
            revision = self._revision(connection)
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('refreshed_at', ?)",
                (now,),
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('revision', ?)",
                (revision,),
            )
            connection.commit()
        return self.status()

    def status(self) -> CatalogStatus:
        if not self.path.is_file():
            return CatalogStatus(self.path.as_posix(), 0, 0, 0, None, None)
        with self._connect() as connection:
            self._create_schema(connection)
            experiments = int(
                connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            )
            records = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
            metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        return CatalogStatus(
            path=self.path.as_posix(),
            experiments=experiments,
            records=records,
            local_records=records,
            refreshed_at=metadata.get("refreshed_at"),
            revision=metadata.get("revision"),
        )

    def experiment_catalog(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            self.refresh()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM experiments ORDER BY experiment_id"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def records(
        self,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("catalog limit must be positive")
        if not self.path.is_file():
            self.refresh()
        clauses: list[str] = []
        values: list[Any] = []
        for key, value in (filters or {}).items():
            if key not in FILTER_FIELDS:
                raise ValueError(f"unsupported catalog filter: {key}")
            if key in {"preset_id", "prompt_id"}:
                clauses.append(f"json_extract(payload, '$.{key}') = ?")
            elif key in {"skill_id", "integration_id"}:
                field = "skill_ids" if key == "skill_id" else "integration_ids"
                clauses.append(
                    "EXISTS (SELECT 1 FROM json_each(json_extract(payload, "
                    f"'$.{field}')) WHERE json_each.value = ?)"
                )
            else:
                clauses.append(f"{key} = ?")
            values.append(str(value))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT payload FROM records{where} ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [json.loads(row[0]) for row in rows]

    def facets(
        self, records: Iterable[Mapping[str, Any]] | None = None
    ) -> dict[str, dict[str, int]]:
        values = list(records) if records is not None else self.records()
        fields = (
            "experiment_id",
            "workload_id",
            "task_name",
            "harness",
            "variant_id",
            "preset_id",
            "prompt_id",
            "context_system_id",
            "provider",
            "model",
            "status",
            "intervention_type",
            "source",
        )
        result = {
            field: dict(
                sorted(
                    Counter(str(row.get(field) or "unknown") for row in values).items()
                )
            )
            for field in fields
        }
        skills: Counter[str] = Counter()
        for row in values:
            selected = row.get("skill_ids") or []
            if isinstance(selected, str):
                selected = [selected]
            if selected:
                skills.update(str(item) for item in selected)
            else:
                skills["none"] += 1
        result["skill_id"] = dict(sorted(skills.items()))
        integrations: Counter[str] = Counter()
        for row in values:
            selected = row.get("integration_ids") or []
            if isinstance(selected, str):
                selected = [selected]
            if selected:
                integrations.update(str(item) for item in selected)
            else:
                integrations["none"] += 1
        result["integration_id"] = dict(sorted(integrations.items()))
        return result

    def read_artifact(self, value: str | Path, *, max_bytes: int = 32_768) -> ArtifactExcerpt:
        if max_bytes < 1 or max_bytes > 131_072:
            raise ValueError("artifact excerpt size must be between 1 and 131072 bytes")
        raw = Path(value)
        path = (raw if raw.is_absolute() else self.repo_root / raw).resolve()
        allowed = [
            (self.repo_root / name).resolve()
            for name in ("jobs", "reports", ".fugue/runtime")
        ]
        if not any(path == root or path.is_relative_to(root) for root in allowed):
            raise ValueError("artifact path is outside Fugue result directories")
        lowered = path.as_posix().lower()
        if any(part in lowered for part in ("/.env", "secret", "credential", "api-key")):
            raise ValueError("artifact path is blocked by the secret policy")
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        truncated = len(data) > max_bytes
        excerpt = data[:max_bytes].decode(errors="replace")
        digest = hashlib.sha256(data).hexdigest()
        return ArtifactExcerpt(
            path=path.relative_to(self.repo_root).as_posix(),
            sha256=digest,
            text=excerpt,
            truncated=truncated,
        )

    def _refresh_local(self, connection: sqlite3.Connection) -> None:
        seen_experiments: set[str] = set()
        variants: dict[tuple[str, str], str] = {}
        experiment_metadata: dict[str, dict[str, Any]] = {}
        for item in list_experiments(self.repo_root):
            experiment = get_experiment(item.id, self.repo_root)
            payload = _experiment_record(experiment, item.sha256)
            seen_experiments.add(experiment.id)
            connection.execute(
                """
                INSERT OR REPLACE INTO experiments(
                    experiment_id, title, config_digest, intervention_type, payload
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.title,
                    item.sha256,
                    payload["intervention_type"],
                    json.dumps(payload, sort_keys=True, default=str),
                ),
            )
            for variant in experiment.variants:
                variants[(experiment.id, variant.id)] = _variant_intervention(variant)
            experiment_metadata[experiment.id] = {
                "manifest": experiment.manifest.as_posix(),
                "experiment_tags": list(experiment.tags),
            }
        if seen_experiments:
            placeholders = ",".join("?" for _ in seen_experiments)
            connection.execute(
                f"DELETE FROM experiments WHERE experiment_id NOT IN ({placeholders})",
                tuple(sorted(seen_experiments)),
            )
        else:
            connection.execute("DELETE FROM experiments")

        rows = export_rows(
            [
                path
                for path in (
                    self.repo_root / "jobs",
                    self.repo_root / ".fugue" / "runtime",
                )
                if path.exists()
            ]
        )
        reports = self.repo_root / "reports"
        if reports.is_dir():
            for path in reports.rglob("*.jsonl"):
                rows.extend(_jsonl(path))
        seen_rows: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not row.get("record_type"):
                continue
            payload = dict(redact_value(row))
            payload["source"] = "local"
            payload["provider"] = payload.get("model_provider") or payload.get("provider")
            key = (str(payload.get("experiment_id") or ""), str(payload.get("variant_id") or ""))
            payload["intervention_type"] = variants.get(key, "unknown")
            payload.update(
                {
                    meta_key: meta_value
                    for meta_key, meta_value in experiment_metadata.get(key[0], {}).items()
                    if not payload.get(meta_key)
                }
            )
            payload["status"] = _record_status(payload)
            row_id = _row_id(payload)
            payload["row_id"] = row_id
            seen_rows.add(row_id)
            self._upsert_record(connection, payload)
        if seen_rows:
            placeholders = ",".join("?" for _ in seen_rows)
            connection.execute(
                f"DELETE FROM records WHERE source = 'local' AND row_id NOT IN ({placeholders})",
                tuple(sorted(seen_rows)),
            )
        else:
            connection.execute("DELETE FROM records WHERE source = 'local'")

    def _upsert_record(self, connection: sqlite3.Connection, payload: dict[str, Any]) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO records(
                row_id, source, record_type, experiment_id, run_id, run_key,
                workload_id, task_name, harness, variant_id, context_system_id,
                provider, model, status, intervention_type, created_at,
                payload
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["row_id"],
                payload.get("source", "local"),
                payload.get("record_type", "trace_only"),
                payload.get("experiment_id"),
                payload.get("run_id"),
                payload.get("run_key"),
                payload.get("workload_id"),
                payload.get("task_name") or payload.get("task_id"),
                payload.get("harness"),
                payload.get("variant_id"),
                payload.get("context_system_id"),
                payload.get("provider"),
                payload.get("model"),
                payload.get("status"),
                payload.get("intervention_type"),
                payload.get("created_at") or payload.get("started_at") or "",
                json.dumps(payload, sort_keys=True, default=str),
            ),
        )

    def _revision(self, connection: sqlite3.Connection) -> str:
        values = connection.execute(
            "SELECT row_id FROM records ORDER BY row_id"
        ).fetchall()
        experiments = connection.execute(
            "SELECT config_digest FROM experiments ORDER BY experiment_id"
        ).fetchall()
        source = "\n".join(str(row[0]) for row in [*experiments, *values])
        return hashlib.sha256(source.encode()).hexdigest()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments(
                experiment_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                config_digest TEXT NOT NULL,
                intervention_type TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records(
                row_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                record_type TEXT NOT NULL,
                experiment_id TEXT,
                run_id TEXT,
                run_key TEXT,
                workload_id TEXT,
                task_name TEXT,
                harness TEXT,
                variant_id TEXT,
                context_system_id TEXT,
                provider TEXT,
                model TEXT,
                status TEXT,
                intervention_type TEXT,
                created_at TEXT,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_records_experiment ON records(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_records_run ON records(run_id);
            CREATE INDEX IF NOT EXISTS idx_records_facets ON records(
                workload_id, harness, variant_id, context_system_id, model
            );
            """
        )


def _experiment_record(experiment: ExperimentSpec, digest: str) -> dict[str, Any]:
    interventions = sorted({_variant_intervention(item) for item in experiment.variants})
    intervention = interventions[0] if len(interventions) == 1 else "mixed"
    return {
        "experiment_id": experiment.id,
        "title": experiment.title,
        "description": experiment.description,
        "manifest": experiment.manifest.as_posix(),
        "model": experiment.model,
        "builder_model": experiment.builder_model,
        "judge_model": experiment.judge_model,
        "tags": experiment.tags,
        "harnesses": experiment.harnesses,
        "variants": [item.to_dict() for item in experiment.variants],
        "workloads": [asdict(item) for item in experiment.workloads],
        "presets": [asdict(item) for item in experiment.presets],
        "config_digest": digest,
        "intervention_type": intervention,
    }


def _variant_intervention(variant: Any) -> str:
    values: list[str] = []
    if variant.prompt_id:
        values.append("prompt")
    if variant.selected_skill_ids:
        values.append("skill")
    if variant.context.system_id != "none":
        values.append("context")
    if variant.integrations:
        values.append("integration")
    if any(
        (
            variant.agent_kwargs,
            variant.agent_env,
            variant.mcp_servers,
            variant.environment,
            variant.verifier,
            variant.retry,
        )
    ):
        values.append("harness_config")
    if not values:
        return "baseline"
    return values[0] if len(values) == 1 else "mixed"


def _row_id(row: Mapping[str, Any]) -> str:
    identity = [
        row.get("source"),
        row.get("record_type"),
        row.get("run_id"),
        row.get("run_key"),
        row.get("cell_id"),
        row.get("task_name") or row.get("task_id"),
        row.get("harness"),
        row.get("variant_id"),
        row.get("trial") or row.get("attempt"),
        row.get("call_id") or row.get("trace_id"),
    ]
    if not any(value not in (None, "") for value in identity[2:]):
        identity.append(json.dumps(row, sort_keys=True, default=str))
    return hashlib.sha256(json.dumps(identity, default=str).encode()).hexdigest()


def _record_status(row: Mapping[str, Any]) -> str:
    if row.get("status"):
        return str(row["status"])
    if row.get("pass") is True:
        return "passed"
    if row.get("pass") is False or row.get("exception_class"):
        return "failed"
    return "unknown"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return values
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values
