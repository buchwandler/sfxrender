"""Machine-readable built-in effect catalog for tools and LLM skills."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .errors import SFXURIError
from .types import SfxSpec
from .uri import parse_sfx_uri

_BUILTIN_CATALOG: dict[str, dict[str, Any]] = {
    "impact.knock": {
        "description": "One or more knocks on a resonant surface.",
        "parameters": {
            "material": {
                "type": "enum",
                "values": ["wood", "oak", "metal", "wall"],
                "default": "wood",
                "description": "Resonant surface being struck.",
            },
            "impactor": {
                "type": "enum",
                "values": ["fingertip", "knuckle", "wooden_object", "metal_object"],
                "default": "knuckle",
                "description": "Object or body part making contact with the surface.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 16,
                "default": 1,
                "description": "Number of audible knocks.",
            },
            "force": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 1.0,
                "default": 0.65,
                "description": "Strength of each knock; lower values are gentler.",
            },
            "interval": {
                "type": "seconds",
                "minimum": 0.08,
                "maximum": 2.0,
                "default": 0.22,
                "description": "Approximate spacing in seconds between repeated knocks.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed. Reuse the same seed "
                    "when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "someone knocks on a door",
                "someone raps or taps on a resonant wall or surface",
                "a small object strikes a resonant surface in a knock-like way",
            ],
            "avoid_when": [
                "a door itself opens or closes",
                "the sound is a heavy crash rather than a knock",
                "footsteps are being described",
            ],
            "examples": [
                "sfx:impact.knock?material=oak&impactor=knuckle&count=3&force=0.65&seed=42",
                "sfx:impact.knock?material=metal&impactor=fingertip&count=2&force=0.35&seed=43",
            ],
        },
    },
    "switch.toggle": {
        "description": "Toggle a small mechanical switch with its actuation and return clicks.",
        "parameters": {
            "state": {
                "type": "enum",
                "values": ["on", "off"],
                "default": "on",
                "description": "Whether the switch toggles into its on or off position.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": ["someone flips a small physical switch", "a mechanical switch clicks on or off"],
            "avoid_when": ["a large power switch or machine startup is intended", "a touchscreen control is tapped"],
            "examples": [
                "sfx:switch.toggle?state=on&seed=42",
                "sfx:switch.toggle?state=off&seed=43",
            ],
        },
    },
    "button.press": {
        "description": "Press a mechanical button and hear its click and return.",
        "parameters": {
            "size": {
                "type": "enum",
                "values": ["small", "large"],
                "default": "small",
                "description": "Approximate size and resonance of the button assembly.",
            },
            "force": {
                "type": "enum",
                "values": ["gentle", "normal", "firm"],
                "default": "normal",
                "description": "How firmly the button is pressed.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": ["a physical button is pressed", "a device button clicks under a finger"],
            "avoid_when": ["a switch is toggled", "a soft screen control is touched"],
            "examples": [
                "sfx:button.press?size=small&force=normal&seed=42",
                "sfx:button.press?size=large&force=firm&seed=43",
            ],
        },
    },
    "object.set_down": {
        "description": "Set a small object onto a wooden or stone surface with a coupled impact.",
        "parameters": {
            "object": {
                "type": "enum",
                "values": ["wood", "metal", "ceramic"],
                "default": "wood",
                "description": "Material character of the object being set down.",
            },
            "surface": {
                "type": "enum",
                "values": ["wood", "stone"],
                "default": "wood",
                "description": "Supporting surface struck by the object.",
            },
            "force": {
                "type": "enum",
                "values": ["gentle", "normal", "firm"],
                "default": "normal",
                "description": "How gently or firmly the object is placed.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": ["a small object is placed on a table", "a metal, wooden, or ceramic object touches down"],
            "avoid_when": ["an object is dropped from a height", "a person sets down a large piece of furniture"],
            "examples": [
                "sfx:object.set_down?object=wood&surface=wood&force=gentle&seed=42",
                "sfx:object.set_down?object=ceramic&surface=stone&force=firm&seed=43",
            ],
        },
    },
    "glass.clink": {
        "description": "Clink wine glasses or tumblers together with a resonant coupled response.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["wine", "tumbler"],
                "default": "wine",
                "description": "Shape and resonant character of the glassware.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
                "default": 1,
                "description": "Number of glass clinks.",
            },
            "force": {
                "type": "enum",
                "values": ["gentle", "normal", "firm"],
                "default": "normal",
                "description": "Strength of each glass contact.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": ["two glasses clink together", "a glass is lightly tapped with a resonant ring"],
            "avoid_when": ["glass shatters", "a glass is set down on a table"],
            "examples": [
                "sfx:glass.clink?style=wine&count=1&force=gentle&seed=42",
                "sfx:glass.clink?style=tumbler&count=2&force=normal&seed=43",
            ],
        },
    },
    "paper.page_turn": {
        "description": "A small stack of sheets turning with papery flutter and light crinkle.",
        "parameters": {
            "pages": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
                "default": 1,
                "description": "Number of sheets turned in sequence.",
            },
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Pace of the page-turn gesture.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": ["turning a page in a book", "flipping a few loose sheets"],
            "avoid_when": ["paper tearing", "a printer feeding or ejecting a sheet"],
            "examples": [
                "sfx:paper.page_turn?pages=1&speed=slow&seed=42",
                "sfx:paper.page_turn?pages=4&speed=fast&seed=43",
            ],
        },
    },
    "paper.handle": {
        "description": "Continuous close-up handling of paper with irregular friction and flutter.",
        "parameters": {
            "duration": {
                "type": "seconds",
                "minimum": 0.3,
                "maximum": 4.0,
                "default": 1.2,
                "description": "Length of the paper-handling gesture.",
            },
            "intensity": {
                "type": "enum",
                "values": ["gentle", "normal", "rough"],
                "default": "normal",
                "description": "Amount of sheet motion and rubbing.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": ["someone shuffles or handles sheets", "a soft paper rustle close by"],
            "avoid_when": ["pages turning one at a time", "a printer mechanism moving paper"],
            "examples": [
                "sfx:paper.handle?duration=1.2&intensity=gentle&seed=42",
                "sfx:paper.handle?duration=2.4&intensity=rough&seed=43",
            ],
        },
    },
    "footsteps.walk": {
        "description": "A sequence of walking footsteps.",
        "parameters": {
            "surface": {
                "type": "enum",
                "values": ["wood", "stone", "gravel", "carpet"],
                "default": "wood",
                "description": "Floor or ground material being walked on.",
            },
            "footwear": {
                "type": "enum",
                "values": ["barefoot", "shoes", "boots", "heels"],
                "default": "shoes",
                "description": "Footwear making contact with the walking surface.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 64,
                "default": 4,
                "description": "Number of audible walking steps.",
            },
            "force": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 1.0,
                "default": 0.6,
                "description": "Perceived loading and strength of the steps.",
            },
            "interval": {
                "type": "seconds",
                "minimum": 0.18,
                "maximum": 2.0,
                "default": 0.52,
                "description": "Approximate spacing in seconds between steps.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed. Reuse the same seed "
                    "when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "a character walks and the footsteps are audible",
                "someone approaches, crosses, or leaves a space on foot",
                "the scene explicitly calls for a sequence of walking steps",
            ],
            "avoid_when": [
                "the character is running",
                "only a single unrelated impact is described",
                "movement is visual or implied but not audibly relevant",
            ],
            "examples": [
                "sfx:footsteps.walk?surface=wood&footwear=boots&count=6&force=0.6&interval=0.52&seed=42",
                "sfx:footsteps.walk?surface=gravel&footwear=shoes&count=4&seed=43",
            ],
        },
    },
    "phone.ring": {
        "description": "A classic or electronic telephone ring.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["classic", "electronic"],
                "default": "classic",
                "description": "Ring character: classic mechanical-style or electronic.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 12,
                "default": 1,
                "description": "Number of audible ring events.",
            },
            "interval": {
                "type": "seconds",
                "minimum": 0.25,
                "maximum": 5.0,
                "default": 1.15,
                "description": "Approximate spacing in seconds between repeated rings.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed. Reuse the same seed "
                    "when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "a telephone audibly rings",
                "a phone begins ringing in a scene",
            ],
            "avoid_when": [
                "a phone merely vibrates",
                "a notification beep or text-message alert is described",
                "a person is already speaking on the phone",
            ],
            "examples": [
                "sfx:phone.ring?style=classic&count=2&seed=42",
                "sfx:phone.ring?style=electronic&count=3&interval=0.72&seed=43",
            ],
        },
    },
    "doorbell.ring": {
        "description": "Activate a mechanical two-note door chime or an electronic doorbell melody.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["chime", "electronic"],
                "default": "chime",
                "description": "Doorbell source: struck mechanical chime bars or electronic tones.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 12,
                "default": 1,
                "description": "Number of doorbell activations.",
            },
            "interval": {
                "type": "seconds",
                "minimum": 0.2,
                "maximum": 10.0,
                "default": 0.8,
                "description": "Spacing in seconds between repeated activations.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative seed identifying the generated chime or tone variation.",
            },
        },
        "llm": {
            "use_when": [
                "a doorbell or entry chime is activated",
                "someone rings a mechanical or electronic doorbell",
            ],
            "avoid_when": [
                "someone knocks on a door",
                "a telephone rings",
                "a phone notification sounds",
            ],
            "examples": [
                "sfx:doorbell.ring?style=chime&count=1&seed=42",
                "sfx:doorbell.ring?style=electronic&count=2&interval=0.8&seed=43",
            ],
        },
    },
    "door.open": {
        "description": "Open a generated door model with latch, hinge friction, and body resonance.",
        "parameters": {
            "material": {
                "type": "enum",
                "values": ["wood", "metal"],
                "default": "wood",
                "description": "Door construction family.",
            },
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Opening motion speed.",
            },
            "creak": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.65,
                "description": "Amount of audible hinge friction or creak from 0 to 1.",
            },
            "force": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 1.0,
                "default": 0.45,
                "description": "Strength of handle, latch, and mechanical excitation during opening.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed that identifies the generated door. "
                    "Reuse the same material and seed for its matching close sound."
                ),
            },
        },
        "llm": {
            "use_when": [
                "a wooden or metal door audibly opens",
                "a character opens a door and latch or hinge motion should be heard",
            ],
            "avoid_when": [
                "the door only closes",
                "someone only knocks on the door",
                "a drawer, cabinet, or printer tray opens",
            ],
            "examples": [
                "sfx:door.open?material=wood&speed=slow&creak=0.7&force=0.45&seed=42",
                "sfx:door.open?material=metal&speed=fast&creak=0.3&seed=43",
            ],
        },
    },
    "door.close": {
        "description": "Close a generated door model with hinge motion, frame impact, and latch catch.",
        "parameters": {
            "material": {
                "type": "enum",
                "values": ["wood", "metal"],
                "default": "wood",
                "description": "Door construction family.",
            },
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Closing motion speed.",
            },
            "creak": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.25,
                "description": "Amount of audible hinge friction or creak during closing.",
            },
            "force": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 1.0,
                "default": 0.65,
                "description": "Strength of the terminal frame and latch impact.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed that identifies the generated door. "
                    "Reuse the same material and seed for its matching open sound."
                ),
            },
        },
        "llm": {
            "use_when": [
                "a wooden or metal door audibly closes",
                "a door shuts and its frame or latch contact should be heard",
            ],
            "avoid_when": [
                "the door is opening",
                "someone only knocks on the door",
                "another kind of drawer or tray closes",
            ],
            "examples": [
                "sfx:door.close?material=wood&speed=normal&creak=0.25&force=0.65&seed=42",
                "sfx:door.close?material=metal&speed=fast&force=0.9&seed=43",
            ],
        },
    },
    "printer.print": {
        "description": "Feed and print one or more sheets through an office printer.",
        "parameters": {
            "pages": {
                "type": "integer",
                "minimum": 1,
                "maximum": 12,
                "default": 1,
                "description": "Number of sheets audibly fed and printed.",
            },
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Mechanical page-feed speed.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed. Reuse the same seed "
                    "when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "an office printer audibly prints one or more sheets",
                "paper is visibly or audibly fed out of a printer",
                "a printer produces a page",
            ],
            "avoid_when": [
                "the printer only displays a message",
                "the printer wakes without printing",
                "the printer performs a reboot or calibration without feeding paper",
            ],
            "examples": [
                "sfx:printer.print?pages=1&speed=normal&seed=42",
                "sfx:printer.print?pages=3&speed=normal&seed=43",
            ],
        },
    },
    "printer.tray_open": {
        "description": "Unlatch and slide open a printer paper tray.",
        "parameters": {
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Speed at which the printer tray slides open.",
            },
            "paper_load": {
                "type": "enum",
                "values": ["empty", "partial", "full"],
                "default": "full",
                "description": "Approximate amount of paper in the tray; controls loose paper movement.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed. Reuse the same seed "
                    "when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "someone opens a printer paper tray",
                "a printer paper drawer unlatches and slides open",
            ],
            "avoid_when": [
                "paper is being printed",
                "a normal room door opens",
                "the tray remains closed while paper is checked visually",
            ],
            "examples": [
                "sfx:printer.tray_open?speed=normal&paper_load=full&seed=42",
                "sfx:printer.tray_open?speed=fast&paper_load=empty&seed=43",
            ],
        },
    },
    "printer.tray_close": {
        "description": "Slide a printer paper tray closed and engage its terminal latch.",
        "parameters": {
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Speed at which the printer tray slides closed.",
            },
            "paper_load": {
                "type": "enum",
                "values": ["empty", "partial", "full"],
                "default": "full",
                "description": "Approximate amount of paper in the tray; controls paper and sliding sounds.",
            },
            "force": {
                "type": "enum",
                "values": ["gentle", "normal", "firm"],
                "default": "normal",
                "description": "How firmly the tray reaches its closed stop.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed. Reuse the same seed "
                    "when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "someone pushes a printer paper tray shut",
                "a printer paper drawer slides closed and latches",
            ],
            "avoid_when": [
                "paper is being printed",
                "a normal room door closes",
                "the tray remains open",
            ],
            "examples": [
                "sfx:printer.tray_close?speed=normal&paper_load=full&force=firm&seed=42",
                "sfx:printer.tray_close?speed=fast&paper_load=empty&force=gentle&seed=43",
            ],
        },
    },
    "printer.power_switch": {
        "description": "Operate a printer power control and its local relay/mechanical response.",
        "parameters": {
            "state": {
                "type": "enum",
                "values": ["off", "on"],
                "default": "off",
                "description": "Whether the audible power action switches the printer off or on.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed for switch and relay sounds. "
                    "Reuse the same seed when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "someone physically switches a printer off",
                "someone physically switches a printer on",
                "a printer power control and local relay response are audible",
            ],
            "avoid_when": [
                "the full printer boot or restart sequence is the intended sound",
                "the printer merely wakes from sleep",
            ],
            "examples": [
                "sfx:printer.power_switch?state=off&seed=42",
                "sfx:printer.power_switch?state=on&seed=42",
            ],
        },
    },
    "printer.restart": {
        "description": "Printer boot and mechanical calibration sequence with an optional confirmation tone.",
        "parameters": {
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Duration and speed of the startup calibration sequence.",
            },
            "beep": {
                "type": "enum",
                "values": ["off", "on"],
                "default": "off",
                "description": "Whether to sound a short electronic confirmation tone after restart.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed for motor and fan sounds. "
                    "Reuse the same seed when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "a printer restarts or boots after power-on",
                "a printer performs a substantial startup or calibration sequence",
            ],
            "avoid_when": [
                "only the power button click is needed",
                "the printer merely wakes from sleep",
                "the printer is actively printing pages",
            ],
            "examples": [
                "sfx:printer.restart?speed=normal&seed=42",
                "sfx:printer.restart?speed=fast&seed=43",
                "sfx:printer.restart?speed=normal&beep=on&seed=44",
            ],
        },
    },
    "pen.write": {
        "description": "A pen writing and scratching across paper.",
        "parameters": {
            "duration": {
                "type": "seconds",
                "minimum": 0.25,
                "maximum": 8.0,
                "default": 1.6,
                "description": "Approximate duration in seconds of continuous handwriting activity.",
            },
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Handwriting stroke speed and cadence.",
            },
            "pressure": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 1.0,
                "default": 0.55,
                "description": "Pen-to-paper contact strength from light to heavy.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed. Reuse the same seed "
                    "when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "a character writes by hand with a pen",
                "pen scratching on paper should be audible",
                "someone scribbles or writes an address or note by hand",
            ],
            "avoid_when": [
                "a pencil is specifically required",
                "someone is typing rather than handwriting",
                "the pen is merely picked up but does not write",
            ],
            "examples": [
                "sfx:pen.write?duration=1.6&speed=normal&pressure=0.55&seed=42",
                "sfx:pen.write?duration=2.4&speed=slow&pressure=0.7&seed=43",
            ],
        },
    },
    "printer.wake": {
        "description": "Wake a sleeping printer with a short relay, motor, fan, and optional confirmation tone.",
        "parameters": {
            "depth": {
                "type": "enum",
                "values": ["light", "deep"],
                "default": "light",
                "description": "Wake response depth; deep has more mechanical activity.",
            },
            "beep": {
                "type": "enum",
                "values": ["off", "on"],
                "default": "off",
                "description": "Whether to sound a short electronic confirmation tone after waking.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": (
                    "Optional non-negative deterministic variation seed. Reuse the same seed "
                    "when reproducible output is required."
                ),
            },
        },
        "llm": {
            "use_when": [
                "a sleeping printer wakes",
                "a printer comes out of standby without a full reboot",
                "a short relay, fan, or motor response marks printer wake-up",
            ],
            "avoid_when": [
                "the printer performs a full power-on restart",
                "paper is actually being printed",
                "only the physical power switch is pressed",
            ],
            "examples": [
                "sfx:printer.wake?depth=light&seed=42",
                "sfx:printer.wake?depth=deep&seed=43",
                "sfx:printer.wake?depth=deep&beep=on&seed=44",
            ],
        },
    },
}


def catalog() -> dict[str, dict[str, Any]]:
    """Return an independent copy of the built-in effect catalog."""

    return deepcopy(_BUILTIN_CATALOG)


_SUPPORTED_PARAMETER_TYPES = {"enum", "integer", "number", "seconds"}


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _validate_parameter_value(
    effect: str, name: str, schema: Mapping[str, Any], value: object
) -> None:
    """Check a default or URI value against one descriptor parameter schema."""
    kind = schema["type"]
    if kind == "enum":
        if not isinstance(value, str):
            raise ValueError(f"parameter {name!r} must be a string enum value")
        allowed = schema["values"]
        if value.lower() not in {choice.lower() for choice in allowed}:
            raise ValueError(
                f"parameter {name!r} must be one of {', '.join(allowed)}, got {value!r}"
            )
        return

    if kind == "integer":
        if isinstance(value, bool):
            raise ValueError(f"parameter {name!r} must be an integer")
        if isinstance(value, str):
            try:
                parsed: int | float = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"parameter {name!r} must be an integer") from exc
        elif isinstance(value, int):
            parsed = value
        else:
            raise ValueError(f"parameter {name!r} must be an integer")
        if name == "seed" and parsed < 0:
            raise ValueError("seed must be a non-negative integer")
    elif kind in {"number", "seconds"}:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"parameter {name!r} must be a number")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"parameter {name!r} must be a number") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"parameter {name!r} must be finite")
    else:
        raise ValueError(f"unsupported parameter type {kind!r}")

    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"parameter {name!r} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"parameter {name!r} must be at most {maximum}")


def validate_effect_descriptor(effect: str, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and defensively copy one built-in or custom effect descriptor."""
    if not isinstance(effect, str) or not effect.strip():
        raise ValueError("effect name must be a non-empty string")
    if not isinstance(descriptor, Mapping):
        raise ValueError("effect descriptor must be a mapping")  # noqa: TRY004

    normalized = deepcopy(dict(descriptor))
    description = normalized.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("descriptor description must be a non-empty string")
    parameters = normalized.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("descriptor parameters must be a mapping")  # noqa: TRY004

    llm = normalized.get("llm")
    if llm is not None and not isinstance(llm, Mapping):
        raise ValueError("descriptor llm metadata must be a mapping")
    llm_metadata: Mapping[str, Any] | None = llm if isinstance(llm, Mapping) else None
    normalized_parameters: dict[str, Any] = {}
    for name, raw_schema in parameters.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("parameter names must be non-empty strings")
        if not isinstance(raw_schema, Mapping):
            raise ValueError(f"parameter {name!r} schema must be a mapping")  # noqa: TRY004
        schema = dict(raw_schema)
        kind = schema.get("type")
        if not isinstance(kind, str) or kind not in _SUPPORTED_PARAMETER_TYPES:
            raise ValueError(
                f"parameter {name!r} has unsupported type {kind!r}; "
                f"expected one of {', '.join(sorted(_SUPPORTED_PARAMETER_TYPES))}"
            )
        parameter_description = schema.get("description")
        if parameter_description is not None and (
            not isinstance(parameter_description, str) or not parameter_description.strip()
        ):
            raise ValueError(f"parameter {name!r} description must be a non-empty string")
        if llm_metadata is not None and parameter_description is None:
            raise ValueError(f"parameter {name!r} requires a description for LLM-visible effects")
        if "required" in schema and not isinstance(schema["required"], bool):
            raise ValueError(f"parameter {name!r} required must be a boolean")

        if kind == "enum":
            values = schema.get("values")
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(
                    f"enum parameter {name!r} requires a non-empty list of non-empty strings"
                )
            if len(set(values)) != len(values):
                raise ValueError(f"enum parameter {name!r} values must be unique")
        elif "values" in schema:
            raise ValueError(f"non-enum parameter {name!r} must not define enum values")

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        for bound_name, bound in (("minimum", minimum), ("maximum", maximum)):
            if bound_name in schema and not _is_finite_number(bound):
                raise ValueError(f"parameter {name!r} {bound_name} must be a finite number")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"parameter {name!r} minimum must not exceed maximum")
        if kind == "enum" and ("minimum" in schema or "maximum" in schema):
            raise ValueError(f"enum parameter {name!r} must not define numeric bounds")

        if "default" in schema:
            default = schema["default"]
            if kind == "enum" and not isinstance(default, str):
                raise ValueError(f"parameter {name!r} default must be a string enum value")
            if kind == "integer" and (isinstance(default, bool) or not isinstance(default, int)):
                raise ValueError(f"parameter {name!r} default must be an integer")
            if kind in {"number", "seconds"} and not _is_finite_number(default):
                raise ValueError(f"parameter {name!r} default must be a finite number")
            try:
                _validate_parameter_value(effect, name, schema, default)
            except ValueError as exc:
                raise ValueError(f"invalid default for parameter {name!r}: {exc}") from exc
        normalized_parameters[name] = schema

    normalized["parameters"] = normalized_parameters
    normalized["parameters"] = normalized_parameters
    if llm_metadata is not None:
        use_when = llm_metadata.get("use_when")
        avoid_when = llm_metadata.get("avoid_when")
        examples = llm_metadata.get("examples")
        if (
            not isinstance(use_when, list)
            or not use_when
            or any(not isinstance(item, str) or not item.strip() for item in use_when)
        ):
            raise ValueError("llm.use_when must be a non-empty list of non-empty strings")
        if not isinstance(avoid_when, list) or any(
            not isinstance(item, str) or not item.strip() for item in avoid_when
        ):
            raise ValueError("llm.avoid_when must be a list of non-empty strings")
        if (
            not isinstance(examples, list)
            or not examples
            or any(not isinstance(item, str) or not item.strip() for item in examples)
        ):
            raise ValueError("llm.examples must be a non-empty list of non-empty strings")

        for index, example in enumerate(examples):
            try:
                spec: SfxSpec = parse_sfx_uri(example)
            except (SFXURIError, TypeError) as exc:
                raise ValueError(f"llm example {index} is not a valid SFX URI: {exc}") from exc
            if spec.effect != effect:
                raise ValueError(
                    f"llm example {index} targets {spec.effect!r}, expected {effect!r}"
                )
            for required_name, required_schema in normalized_parameters.items():
                if required_schema.get("required", False) and required_name not in spec.parameters:
                    raise ValueError(
                        f"llm example {index} omits required parameter {required_name!r}"
                    )
            for parameter_name, value in spec.parameters.items():
                example_schema = normalized_parameters.get(parameter_name)
                if example_schema is None:
                    raise ValueError(
                        f"llm example {index} has unknown parameter {parameter_name!r}"
                    )
                try:
                    _validate_parameter_value(effect, parameter_name, example_schema, value)
                except ValueError as exc:
                    raise ValueError(f"invalid llm example {index}: {exc}") from exc

        normalized["llm"] = deepcopy(dict(llm_metadata))
    return normalized


_LLM_RULES = (
    "Use only effects listed in effects.",
    "Use only parameters listed for the selected effect.",
    "Respect enum choices and numeric bounds.",
    "If no listed effect matches the audible event, do not invent an effect.",
    "Prefer omitting an SFX over emitting an invalid URI.",
    "Use a fixed non-negative seed when reproducible rendering is desired.",
    "Use a short sound description in an SSMD audio span, not replacement dialogue.",
)


def build_llm_catalog(
    effects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic, defensive LLM authoring manifest from descriptors."""
    authorable: dict[str, dict[str, Any]] = {}
    for effect in sorted(effects):
        descriptor = effects[effect]
        llm = descriptor.get("llm")
        if not isinstance(llm, Mapping):
            continue
        use_when = llm.get("use_when")
        examples = llm.get("examples")
        if (
            not isinstance(descriptor.get("description"), str)
            or not isinstance(descriptor.get("parameters"), Mapping)
            or not isinstance(use_when, list)
            or not use_when
            or not isinstance(examples, list)
            or not examples
        ):
            continue
        authorable[effect] = deepcopy(dict(descriptor))

    return {
        "schema": "sfxrender.llm-catalog.v1",
        "uri_template": "sfx:<effect>?<key>=<value>&...",
        "ssmd_template": '[short sound description]{src="sfx:<effect>?<query>"}',
        "rules": list(_LLM_RULES),
        "effects": authorable,
    }


def llm_catalog() -> dict[str, Any]:
    """Return the built-in effects as a model-neutral LLM authoring manifest."""
    return build_llm_catalog(catalog())
