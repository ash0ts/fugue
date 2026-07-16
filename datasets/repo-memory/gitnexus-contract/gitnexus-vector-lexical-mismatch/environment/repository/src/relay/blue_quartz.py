"""Select healthy workers for new assignments."""


def available_workers(health: dict[str, bool]) -> tuple[str, ...]:
    return tuple(name for name, ready in health.items() if ready)
