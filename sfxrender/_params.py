"""Small parameter helpers used by built-in effects."""

from __future__ import annotations

from collections.abc import Mapping


def integer(params: Mapping[str, str], name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = params.get(name)
    value = default if raw is None else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def number(
    params: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = params.get(name)
    value = default if raw is None else float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def choice(params: Mapping[str, str], name: str, default: str, allowed: set[str]) -> str:
    value = params.get(name, default).lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return value
