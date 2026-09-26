"""Small parameter helpers used by built-in effects."""

from __future__ import annotations

import math
from collections.abc import Mapping


def integer(
    params: Mapping[str, str], name: str, default: int, *, minimum: int, maximum: int
) -> int:
    raw = params.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
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
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def choice(params: Mapping[str, str], name: str, default: str, allowed: set[str]) -> str:
    value = params.get(name, default).lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return value
