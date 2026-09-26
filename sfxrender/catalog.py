"""Machine-readable built-in effect catalog for tools and LLM skills."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_BUILTIN_CATALOG: dict[str, dict[str, Any]] = {
    "impact.knock": {
        "description": "One or more knocks on a resonant surface.",
        "parameters": {
            "material": {"type": "enum", "values": ["wood", "oak", "metal", "wall"], "default": "wood"},
            "count": {"type": "integer", "minimum": 1, "maximum": 16, "default": 1},
            "force": {"type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.65},
            "interval": {"type": "seconds", "minimum": 0.08, "maximum": 2.0, "default": 0.22},
            "seed": {"type": "integer", "required": False},
        },
    },
    "footsteps.walk": {
        "description": "A sequence of walking footsteps.",
        "parameters": {
            "surface": {"type": "enum", "values": ["wood", "stone", "gravel", "carpet"], "default": "wood"},
            "footwear": {"type": "enum", "values": ["barefoot", "shoes", "boots", "heels"], "default": "shoes"},
            "count": {"type": "integer", "minimum": 1, "maximum": 64, "default": 4},
            "force": {"type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.6},
            "interval": {"type": "seconds", "minimum": 0.18, "maximum": 2.0, "default": 0.52},
            "seed": {"type": "integer", "required": False},
        },
    },
    "phone.ring": {
        "description": "A classic or electronic telephone ring.",
        "parameters": {
            "style": {"type": "enum", "values": ["classic", "electronic"], "default": "classic"},
            "count": {"type": "integer", "minimum": 1, "maximum": 12, "default": 1},
            "interval": {"type": "seconds", "minimum": 0.25, "maximum": 5.0, "default": 1.15},
            "seed": {"type": "integer", "required": False},
        },
    },
    "door.open": {
        "description": "A simple procedural door opening/creak.",
        "parameters": {
            "material": {"type": "enum", "values": ["wood", "metal"], "default": "wood"},
            "speed": {"type": "enum", "values": ["slow", "normal", "fast"], "default": "normal"},
            "creak": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.65},
            "seed": {"type": "integer", "required": False},
        },
    },
}


def catalog() -> dict[str, dict[str, Any]]:
    """Return an independent copy of the built-in effect catalog."""

    return deepcopy(_BUILTIN_CATALOG)
