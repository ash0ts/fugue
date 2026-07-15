from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any


async def run_job(config_path: Path) -> tuple[str, Path]:
    try:
        from harbor import Job, JobConfig
    except ImportError as exc:  # pragma: no cover - exercised in the image
        raise RuntimeError("Harbor is not installed; install fugue[serve]") from exc

    config = JobConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    job = await Job.create(config)
    result = await job.run()
    if result.stats.n_errored_trials or result.stats.n_cancelled_trials:
        raise RuntimeError(
            "Harbor request failed: "
            f"{result.stats.n_errored_trials} errored, "
            f"{result.stats.n_cancelled_trials} cancelled"
        )
    return extract_final_answer(result, job.job_dir), job.job_dir


def extract_final_answer(result: Any, job_dir: Path) -> str:
    candidates: list[str] = []
    for trial in getattr(result, "trial_results", None) or []:
        context = getattr(trial, "agent_result", None)
        metadata = getattr(context, "metadata", None) if context else None
        candidates.extend(_answer_candidates(metadata))
    for path in sorted(
        [*job_dir.rglob("*.json"), *job_dir.rglob("*.jsonl")],
        key=lambda item: (item.stat().st_mtime_ns, item.as_posix()),
    ):
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    candidates.extend(_answer_candidates(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    values = [value.strip() for value in candidates if value and value.strip()]
    if not values:
        raise RuntimeError("Harbor completed without a trustworthy final answer")
    return values[-1]


def _answer_candidates(value: Any) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_answer_candidates(item))
        return result
    if not isinstance(value, dict):
        return []
    result: list[str] = []
    role = str(value.get("role") or "").lower()
    source = str(value.get("source") or "").lower()
    kind = str(value.get("type") or "").lower()
    if role == "assistant" or kind in {
        "agent_message",
        "assistant_message",
        "output_text",
    }:
        result.extend(_text_values(value.get("content") or value.get("text")))
    if source == "agent":
        result.extend(_text_values(value.get("message")))
    for key in ("final_answer", "output_text", "finalAssistantVisibleText"):
        if key in value:
            result.extend(_text_values(value[key]))
    item = value.get("item")
    if isinstance(item, dict) and str(item.get("type") or "") == "agent_message":
        result.extend(_text_values(item.get("text") or item.get("content")))
    for key, item_value in value.items():
        if key not in {
            "content",
            "text",
            "message",
            "final_answer",
            "output_text",
            "finalAssistantVisibleText",
            "item",
        }:
            result.extend(_answer_candidates(item_value))
    return result


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    result.append(text)
        return result
    if isinstance(value, dict):
        return _text_values(value.get("text") or value.get("content"))
    return []


async def _main(config_path: Path, result_path: Path) -> int:
    task = asyncio.create_task(run_job(config_path))
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, task.cancel)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    try:
        answer, _ = await task
        payload = {"status": "completed", "answer": answer}
        code = 0
    except asyncio.CancelledError:
        payload = {"status": "cancelled", "error": "request cancelled"}
        code = 130
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        code = 1
    result_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    return code


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 2:
        raise SystemExit("usage: python -m fugue.serve.worker CONFIG RESULT")
    return asyncio.run(_main(Path(values[0]), Path(values[1])))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
