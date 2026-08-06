from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fugue.bench.candidates import stable_digest

THREE_WORKER_CPU_COUNT = 8
THREE_WORKER_AVAILABLE_MEMORY_GIB = 16
THREE_WORKER_FREE_DISK_GIB = 30

# Two 2 CPU/4 GiB Harbor cells need this smaller, still fail-closed tier. The
# disk floor reserves room for task overlays, two writable layers, and evidence.
TWO_WORKER_CPU_COUNT = 4
TWO_WORKER_AVAILABLE_MEMORY_GIB = 8
TWO_WORKER_FREE_DISK_GIB = 15

CapacityClass = Literal["below-two-worker", "two-worker", "three-worker"]


class HostCapacityUnavailableError(RuntimeError):
    """The host cannot safely execute the approved admission tier."""


@dataclass(frozen=True)
class HostCapacityObservationV1:
    """Injectable, non-persisted host measurements used to select a tier."""

    cpu_count: int
    available_memory_gib: float
    free_disk_gib: float

    def __post_init__(self) -> None:
        if self.cpu_count < 1:
            raise ValueError("host capacity CPU count must be positive")
        if self.available_memory_gib < 0 or self.free_disk_gib < 0:
            raise ValueError("host capacity memory and disk must be non-negative")


HostCapacityProbe = Callable[[Path], HostCapacityObservationV1]


@dataclass(frozen=True)
class HostCapacityReceiptV1:
    """Stable, digest-bound capacity class selected for one pure preview.

    Exact free memory and disk values fluctuate continuously. Persisting them
    would make an otherwise identical preview change from one command to the
    next. The receipt therefore binds threshold classes and the exact selected
    admission requirements. Execution probes fresh measurements and verifies
    that the approved class remains satisfied before admitting any cells.
    """

    schema_version: Literal[1]
    kind: Literal["host-capacity-receipt"]
    probe_revision: Literal["host-capacity-v1"]
    selected_worker_limit: Literal[2, 3]
    required_cpu_count: int
    required_available_memory_gib: int
    required_free_disk_gib: int
    cpu_capacity_class: CapacityClass
    memory_capacity_class: CapacityClass
    disk_capacity_class: CapacityClass
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported host capacity receipt schema")
        if self.kind != "host-capacity-receipt":
            raise ValueError("host capacity receipt kind is invalid")
        if self.probe_revision != "host-capacity-v1":
            raise ValueError("host capacity probe revision is invalid")
        if self.selected_worker_limit not in {2, 3}:
            raise ValueError("host capacity worker limit must be two or three")
        expected = _requirements(self.selected_worker_limit)
        if (
            self.required_cpu_count,
            self.required_available_memory_gib,
            self.required_free_disk_gib,
        ) != expected:
            raise ValueError("host capacity requirements do not match its tier")
        classes = (
            self.cpu_capacity_class,
            self.memory_capacity_class,
            self.disk_capacity_class,
        )
        if any(
            value not in {"below-two-worker", "two-worker", "three-worker"}
            for value in classes
        ):
            raise ValueError("host capacity class is invalid")
        minimum_class = "three-worker" if self.selected_worker_limit == 3 else "two-worker"
        if any(_capacity_rank(value) < _capacity_rank(minimum_class) for value in classes):
            raise ValueError("host capacity classes do not satisfy the selected tier")
        if self.selected_worker_limit == 2 and all(
            value == "three-worker" for value in classes
        ):
            raise ValueError("two-worker fallback is invalid on a three-worker host")
        digest = stable_digest(self._unsigned())
        if self.receipt_digest and self.receipt_digest != digest:
            raise ValueError("host capacity receipt digest does not match")
        object.__setattr__(self, "receipt_digest", digest)

    @classmethod
    def from_observation(
        cls, observation: HostCapacityObservationV1
    ) -> HostCapacityReceiptV1:
        classes: tuple[CapacityClass, CapacityClass, CapacityClass] = (
            _capacity_class(
                observation.cpu_count,
                two=TWO_WORKER_CPU_COUNT,
                three=THREE_WORKER_CPU_COUNT,
            ),
            _capacity_class(
                observation.available_memory_gib,
                two=TWO_WORKER_AVAILABLE_MEMORY_GIB,
                three=THREE_WORKER_AVAILABLE_MEMORY_GIB,
            ),
            _capacity_class(
                observation.free_disk_gib,
                two=TWO_WORKER_FREE_DISK_GIB,
                three=THREE_WORKER_FREE_DISK_GIB,
            ),
        )
        if all(value == "three-worker" for value in classes):
            workers: Literal[2, 3] = 3
        elif all(_capacity_rank(value) >= _capacity_rank("two-worker") for value in classes):
            workers = 2
        else:
            failed = [
                label
                for label, value in zip(("CPU", "memory", "disk"), classes, strict=True)
                if value == "below-two-worker"
            ]
            raise HostCapacityUnavailableError(
                "host capacity is below the two-worker safety floor: "
                + ", ".join(failed)
            )
        required = _requirements(workers)
        return cls(
            schema_version=1,
            kind="host-capacity-receipt",
            probe_revision="host-capacity-v1",
            selected_worker_limit=workers,
            required_cpu_count=required[0],
            required_available_memory_gib=required[1],
            required_free_disk_gib=required[2],
            cpu_capacity_class=classes[0],
            memory_capacity_class=classes[1],
            disk_capacity_class=classes[2],
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> HostCapacityReceiptV1:
        known = {
            "schema_version",
            "kind",
            "probe_revision",
            "selected_worker_limit",
            "required_cpu_count",
            "required_available_memory_gib",
            "required_free_disk_gib",
            "cpu_capacity_class",
            "memory_capacity_class",
            "disk_capacity_class",
            "receipt_digest",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                "unknown host capacity receipt field(s): " + ", ".join(unknown)
            )
        return cls(
            schema_version=_strict_int(raw.get("schema_version"), "schema_version"),
            kind=str(raw.get("kind") or ""),  # type: ignore[arg-type]
            probe_revision=str(raw.get("probe_revision") or ""),  # type: ignore[arg-type]
            selected_worker_limit=_strict_int(
                raw.get("selected_worker_limit"), "selected_worker_limit"
            ),  # type: ignore[arg-type]
            required_cpu_count=_strict_int(
                raw.get("required_cpu_count"), "required_cpu_count"
            ),
            required_available_memory_gib=_strict_int(
                raw.get("required_available_memory_gib"),
                "required_available_memory_gib",
            ),
            required_free_disk_gib=_strict_int(
                raw.get("required_free_disk_gib"), "required_free_disk_gib"
            ),
            cpu_capacity_class=str(raw.get("cpu_capacity_class") or ""),  # type: ignore[arg-type]
            memory_capacity_class=str(raw.get("memory_capacity_class") or ""),  # type: ignore[arg-type]
            disk_capacity_class=str(raw.get("disk_capacity_class") or ""),  # type: ignore[arg-type]
            receipt_digest=str(raw.get("receipt_digest") or ""),
        )

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "probe_revision": self.probe_revision,
            "selected_worker_limit": self.selected_worker_limit,
            "required_cpu_count": self.required_cpu_count,
            "required_available_memory_gib": self.required_available_memory_gib,
            "required_free_disk_gib": self.required_free_disk_gib,
            "cpu_capacity_class": self.cpu_capacity_class,
            "memory_capacity_class": self.memory_capacity_class,
            "disk_capacity_class": self.disk_capacity_class,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._unsigned(), "receipt_digest": self.receipt_digest}


def select_host_capacity_receipt(
    root: Path, *, probe: HostCapacityProbe | None = None
) -> HostCapacityReceiptV1:
    return HostCapacityReceiptV1.from_observation(
        (probe or probe_host_capacity)(root)
    )


def verify_host_capacity_receipt(
    receipt: HostCapacityReceiptV1,
    root: Path,
    *,
    probe: HostCapacityProbe | None = None,
) -> HostCapacityObservationV1:
    """Recheck the approved tier without silently selecting a new schedule."""

    observation = (probe or probe_host_capacity)(root)
    requirements = _requirements(receipt.selected_worker_limit)
    failures = []
    if observation.cpu_count < requirements[0]:
        failures.append(f"CPU {observation.cpu_count} < {requirements[0]}")
    if observation.available_memory_gib < requirements[1]:
        failures.append(
            "available memory "
            f"{observation.available_memory_gib:.1f} GiB < {requirements[1]} GiB"
        )
    if observation.free_disk_gib < requirements[2]:
        failures.append(
            f"free disk {observation.free_disk_gib:.1f} GiB < {requirements[2]} GiB"
        )
    if failures:
        raise HostCapacityUnavailableError(
            "approved host-capacity tier is no longer satisfied; no cells were "
            "admitted: "
            + "; ".join(failures)
        )
    return observation


def probe_host_capacity(root: Path) -> HostCapacityObservationV1:
    """Read host CPU, available memory, and filesystem free space without writes."""

    cpu_count = os.cpu_count() or 0
    memory_bytes = _available_memory_bytes()
    disk_bytes = shutil.disk_usage(root.resolve()).free
    gib = 1024**3
    return HostCapacityObservationV1(
        cpu_count=cpu_count,
        available_memory_gib=memory_bytes / gib,
        free_disk_gib=disk_bytes / gib,
    )


def _available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(
            r"^MemAvailable:\s+(\d+)\s+kB$",
            meminfo.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if match:
            return int(match.group(1)) * 1024
    if sys.platform == "darwin":
        completed = subprocess.run(  # noqa: S603 - fixed read-only host command
            ("vm_stat",),
            check=True,
            capture_output=True,
            text=True,
        )
        page_match = re.search(r"page size of (\d+) bytes", completed.stdout)
        if not page_match:
            raise HostCapacityUnavailableError("vm_stat did not report a page size")
        available_pages = 0
        for label in (
            "Pages free",
            "Pages inactive",
            "Pages speculative",
            "Pages purgeable",
        ):
            match = re.search(rf"^{label}:\s+(\d+)\.$", completed.stdout, re.MULTILINE)
            if match:
                available_pages += int(match.group(1))
        if available_pages:
            return available_pages * int(page_match.group(1))
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise HostCapacityUnavailableError(
            "host available memory could not be measured"
        ) from exc
    if page_size <= 0 or available_pages < 0:
        raise HostCapacityUnavailableError("host available memory is invalid")
    return page_size * available_pages


def _requirements(worker_limit: int) -> tuple[int, int, int]:
    if worker_limit == 3:
        return (
            THREE_WORKER_CPU_COUNT,
            THREE_WORKER_AVAILABLE_MEMORY_GIB,
            THREE_WORKER_FREE_DISK_GIB,
        )
    if worker_limit == 2:
        return (
            TWO_WORKER_CPU_COUNT,
            TWO_WORKER_AVAILABLE_MEMORY_GIB,
            TWO_WORKER_FREE_DISK_GIB,
        )
    raise ValueError("host capacity worker limit must be two or three")


def _capacity_class(value: float, *, two: float, three: float) -> CapacityClass:
    if value >= three:
        return "three-worker"
    if value >= two:
        return "two-worker"
    return "below-two-worker"


def _capacity_rank(value: CapacityClass) -> int:
    return {
        "below-two-worker": 0,
        "two-worker": 1,
        "three-worker": 2,
    }[value]


def _strict_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"host capacity {label} must be an integer")
    return value
