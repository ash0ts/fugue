from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable


@dataclass(frozen=True)
class DistributionAsset:
    """One immutable build input loaded from the installed distribution."""

    name: str
    body: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    def read_bytes(self) -> bytes:
        """Match the read-only subset of pathlib.Path used by callers/tests."""

        return self.body

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding)


def runtime_assets(component_id: str) -> tuple[DistributionAsset, ...]:
    """Return checkout-independent build assets for one managed component."""

    root = files("fugue").joinpath("resources", "runtime", component_id)
    if not root.is_dir():
        return ()
    return tuple(
        _read_asset(item)
        for item in sorted(root.iterdir(), key=lambda candidate: candidate.name)
        if item.is_file()
    )


def vendor_asset(name: str) -> DistributionAsset:
    """Return a pinned vendor archive from the installed distribution."""

    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("vendor asset name must be a safe basename")
    item = files("fugue").joinpath("resources", "vendor", name)
    if not item.is_file():
        raise FileNotFoundError(f"packaged Fugue vendor asset is missing: {name}")
    return _read_asset(item)


def fugue_package_asset(name: str) -> DistributionAsset:
    """Return a file owned by the installed Fugue Python package."""

    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("Fugue package asset name must be a safe basename")
    item = files("fugue").joinpath(name)
    if not item.is_file():
        raise FileNotFoundError(f"packaged Fugue asset is missing: {name}")
    return _read_asset(item)


def _read_asset(item: Traversable) -> DistributionAsset:
    return DistributionAsset(name=item.name, body=item.read_bytes())
