"""Withdraw descendant allocations after an ancestor claim expires."""

from collections.abc import Iterable


def withdraw_descendants(parent: str, lineage: dict[str, list[str]]) -> tuple[str, ...]:
    """Return every child allocation that must be withdrawn, deepest first."""
    pending = list(lineage.get(parent, ()))
    withdrawn: list[str] = []
    while pending:
        child = pending.pop()
        withdrawn.append(child)
        pending.extend(lineage.get(child, ()))
    return tuple(withdrawn)


def preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
