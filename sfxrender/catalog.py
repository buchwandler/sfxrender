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
            "use_when": [
                "someone flips a small physical switch",
                "a mechanical switch clicks on or off",
            ],
            "avoid_when": [
                "a large power switch or machine startup is intended",
                "a touchscreen control is tapped",
            ],
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
            "use_when": [
                "a small object is placed on a table",
                "a metal, wooden, or ceramic object touches down",
            ],
            "avoid_when": [
                "an object is dropped from a height",
                "a person sets down a large piece of furniture",
            ],
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
            "use_when": [
                "two glasses clink together",
                "a glass is lightly tapped with a resonant ring",
            ],
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
        "description": "A sequence of walking footsteps with pace-sensitive heel, sole, and toe contact articulation.",
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
    "footsteps.run": {
        "description": "A sequence of running steps with stronger loading, shorter contact articulation, and pace-sensitive scuffs.",
        "parameters": {
            "surface": {
                "type": "enum",
                "values": ["wood", "stone", "gravel", "carpet"],
                "default": "wood",
                "description": "Existing floor or ground material underfoot.",
            },
            "footwear": {
                "type": "enum",
                "values": ["barefoot", "shoes", "boots", "heels"],
                "default": "shoes",
                "description": "Footwear making contact with the running surface.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 64,
                "default": 8,
                "description": "Number of running steps.",
            },
            "force": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 1.0,
                "default": 0.68,
                "description": "Effort and loading of the run.",
            },
            "interval": {
                "type": "seconds",
                "minimum": 0.2,
                "maximum": 0.8,
                "default": 0.32,
                "description": "Approximate spacing between running steps.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a character runs across an audible surface",
                "quick footfalls approach or recede",
            ],
            "avoid_when": ["a steady walking pace", "steps climbing or descending stairs"],
            "examples": [
                "sfx:footsteps.run?surface=wood&footwear=shoes&count=8&force=0.68&interval=0.32&seed=42",
                "sfx:footsteps.run?surface=gravel&footwear=boots&count=6&seed=43",
            ],
        },
    },
    "footsteps.stairs": {
        "description": "Footsteps climbing or descending stairs with direction-sensitive forefoot or landing emphasis.",
        "parameters": {
            "surface": {
                "type": "enum",
                "values": ["wood", "stone"],
                "default": "wood",
                "description": "Material of the individual stair treads.",
            },
            "footwear": {
                "type": "enum",
                "values": ["barefoot", "shoes", "boots", "heels"],
                "default": "shoes",
                "description": "Footwear making contact with each tread.",
            },
            "direction": {
                "type": "enum",
                "values": ["up", "down"],
                "default": "up",
                "description": "Whether the steps ascend or descend the stairs.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 64,
                "default": 5,
                "description": "Number of stair steps.",
            },
            "force": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 1.0,
                "default": 0.62,
                "description": "Effort and loading on the stair treads.",
            },
            "interval": {
                "type": "seconds",
                "minimum": 0.28,
                "maximum": 1.6,
                "default": 0.58,
                "description": "Approximate spacing between stair steps.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": ["a person climbs stairs", "footsteps descend a flight of stairs"],
            "avoid_when": ["walking across a flat floor", "running on level ground"],
            "examples": [
                "sfx:footsteps.stairs?surface=wood&direction=up&count=6&seed=42",
                "sfx:footsteps.stairs?surface=stone&direction=down&footwear=boots&count=5&seed=43",
            ],
        },
    },
    "electronics.hum": {
        "description": "A quiet harmonic electrical hum from a transformer or appliance.",
        "parameters": {
            "source": {
                "type": "enum",
                "values": ["transformer", "appliance"],
                "default": "transformer",
                "description": "Electrical source character and harmonic balance.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.5,
                "maximum": 8.0,
                "default": 2.0,
                "description": "Length of the continuous hum.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a quiet electrical transformer or appliance hums",
                "a powered device has a faint harmonic buzz",
            ],
            "avoid_when": [
                "a distinct alert or notification tone",
                "a motor or engine is the main sound",
            ],
            "examples": [
                "sfx:electronics.hum?source=transformer&duration=2.0&seed=42",
                "sfx:electronics.hum?source=appliance&duration=4.0&seed=43",
            ],
        },
    },
    "device.beep": {
        "description": "A short electronic confirmation or alert beep pattern.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["soft", "alert"],
                "default": "soft",
                "description": "Whether the beep is gentle or attention-getting.",
            },
            "pattern": {
                "type": "enum",
                "values": ["single", "double", "triple"],
                "default": "single",
                "description": "Number of short beeps in the pattern.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a generic device gives a short beep",
                "a control emits a soft or urgent confirmation tone",
            ],
            "avoid_when": [
                "a phone plays a multi-note notification",
                "a sustained alarm or ringing sound",
            ],
            "examples": [
                "sfx:device.beep?style=soft&pattern=single&seed=42",
                "sfx:device.beep?style=alert&pattern=triple&seed=43",
            ],
        },
    },
    "phone.notification": {
        "description": "A compact multi-note smartphone notification chime.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["gentle", "urgent"],
                "default": "gentle",
                "description": "Pitch contour and urgency of the notification.",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
                "default": 1,
                "description": "Number of notification chime repetitions.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a smartphone receives a message or app alert",
                "a short personal notification chime sounds",
            ],
            "avoid_when": ["a generic single device beep", "a telephone ringing repeatedly"],
            "examples": [
                "sfx:phone.notification?style=gentle&count=1&seed=42",
                "sfx:phone.notification?style=urgent&count=2&seed=43",
            ],
        },
    },
    "phone.vibrate": {
        "description": "A phone's eccentric motor buzz transmitted through a wooden or stone surface.",
        "parameters": {
            "duration": {
                "type": "seconds",
                "minimum": 0.2,
                "maximum": 5.0,
                "default": 0.8,
                "description": "Length of the vibration event.",
            },
            "intensity": {
                "type": "enum",
                "values": ["gentle", "normal", "strong"],
                "default": "normal",
                "description": "Strength of the vibrating motor and surface response.",
            },
            "pattern": {
                "type": "enum",
                "values": ["steady", "pulsed"],
                "default": "steady",
                "description": "Whether vibration runs continuously or in short bursts.",
            },
            "surface": {
                "type": "enum",
                "values": ["wood", "stone"],
                "default": "wood",
                "description": "Surface coupling the phone's motor vibration.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a phone vibrates on a table",
                "an incoming call or notification is felt as a surface buzz",
            ],
            "avoid_when": [
                "a phone plays an audible notification melody",
                "a large appliance motor runs",
            ],
            "examples": [
                "sfx:phone.vibrate?duration=0.8&intensity=normal&pattern=steady&surface=wood&seed=42",
                "sfx:phone.vibrate?duration=1.4&intensity=strong&pattern=pulsed&surface=stone&seed=43",
            ],
        },
    },
    "device.power_on": {
        "description": "A generic small device powers on with a switch contact, rising motor and hum, then confirmation beep.",
        "parameters": {
            "device": {
                "type": "enum",
                "values": ["small", "appliance"],
                "default": "small",
                "description": "Size and startup character of the powered device.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a generic appliance or small device switches on",
                "a device starts with a brief powered-up tone",
            ],
            "avoid_when": [
                "printer-specific startup or wake behavior",
                "an engine starting or a phone notification",
            ],
            "examples": [
                "sfx:device.power_on?device=small&seed=42",
                "sfx:device.power_on?device=appliance&seed=43",
            ],
        },
    },
    "device.power_off": {
        "description": "A generic device powers off as its motor and electrical hum decay into a switch contact.",
        "parameters": {
            "device": {
                "type": "enum",
                "values": ["small", "appliance"],
                "default": "small",
                "description": "Size and shutdown character of the powered device.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a generic powered device switches off",
                "an appliance motor and hum wind down",
            ],
            "avoid_when": [
                "printer-specific shutdown behavior",
                "a device powers up or emits a notification",
            ],
            "examples": [
                "sfx:device.power_off?device=small&seed=42",
                "sfx:device.power_off?device=appliance&seed=43",
            ],
        },
    },
    "room_tone": {
        "description": "A finite quiet indoor ambience bed with soft air, low noise, and optional electrical hum.",
        "parameters": {
            "character": {
                "type": "enum",
                "values": ["quiet", "ventilated", "electrical"],
                "default": "quiet",
                "description": "Background room character.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.5,
                "maximum": 30.0,
                "default": 4.0,
                "description": "Length of the finite room-tone recording.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a quiet indoor room has audible air or electrical noise",
                "a scene needs a low-level finite room bed",
            ],
            "avoid_when": [
                "distinct office activity or city traffic is audible",
                "intelligible speech or music is required",
            ],
            "examples": [
                "sfx:room_tone?character=quiet&duration=4.0&seed=42",
                "sfx:room_tone?character=electrical&duration=6.0&seed=43",
            ],
        },
    },
    "office.ambience": {
        "description": "A finite office room bed with sparse or busy nonverbal clicks and rustles.",
        "parameters": {
            "activity": {
                "type": "enum",
                "values": ["quiet", "busy"],
                "default": "quiet",
                "description": "Density of distant office micro-events.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 2.0,
                "maximum": 30.0,
                "default": 8.0,
                "description": "Length of the finite office ambience.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a quiet or active office is audible in the background",
                "subtle distant office clicks and room noise are needed",
            ],
            "avoid_when": [
                "a close printer action is the focus",
                "intelligible conversation or music is required",
            ],
            "examples": [
                "sfx:office.ambience?activity=quiet&duration=8.0&seed=42",
                "sfx:office.ambience?activity=busy&duration=12.0&seed=43",
            ],
        },
    },
    "city.ambience": {
        "description": "A finite urban traffic bed with seeded distant vehicle swells and no foreground perspective shift.",
        "parameters": {
            "activity": {
                "type": "enum",
                "values": ["calm", "busy"],
                "default": "calm",
                "description": "Density of distant traffic movement.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 4.0,
                "maximum": 30.0,
                "default": 12.0,
                "description": "Length of the finite city ambience.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "distant city traffic provides an outdoor background",
                "an urban street ambience is needed without a close vehicle pass",
            ],
            "avoid_when": [
                "a specific car passing is the focal sound",
                "foreground dialogue or music is required",
            ],
            "examples": [
                "sfx:city.ambience?activity=calm&duration=12.0&seed=42",
                "sfx:city.ambience?activity=busy&duration=16.0&seed=43",
            ],
        },
    },
    "wind": {
        "description": "A finite filtered wind wash with slow gust variation and an optional leafy high-frequency texture.",
        "parameters": {
            "intensity": {
                "type": "enum",
                "values": ["light", "strong"],
                "default": "light",
                "description": "Strength and density of the wind.",
            },
            "texture": {
                "type": "enum",
                "values": ["smooth", "leafy"],
                "default": "smooth",
                "description": "Whether the wind is smooth or carries light foliage-like hiss.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 1.0,
                "maximum": 30.0,
                "default": 6.0,
                "description": "Length of the finite wind recording.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "wind moves steadily or gusts in the background",
                "a soft airy outdoor wind bed is needed",
            ],
            "avoid_when": [
                "rainfall or fire crackle is the focus",
                "a designed transition sweep is required",
            ],
            "examples": [
                "sfx:wind?intensity=light&texture=smooth&duration=6.0&seed=42",
                "sfx:wind?intensity=strong&texture=leafy&duration=8.0&seed=43",
            ],
        },
    },
    "transition.whoosh": {
        "description": "A short finite turbulent transition sweep with a rising or falling spectral brightness.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["soft", "forceful"],
                "default": "soft",
                "description": "Overall weight of the whoosh.",
            },
            "direction": {
                "type": "enum",
                "values": ["rise", "fall"],
                "default": "rise",
                "description": "Whether the sweep brightens or darkens through its duration.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.25,
                "maximum": 4.0,
                "default": 1.2,
                "description": "Length of the transition sweep.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a brief designed transition moves into or out of a scene",
                "a short non-tonal cinematic whoosh is needed",
            ],
            "avoid_when": [
                "continuous outdoor wind is required",
                "a vehicle or other specific object passes by",
            ],
            "examples": [
                "sfx:transition.whoosh?style=soft&direction=rise&duration=1.2&seed=42",
                "sfx:transition.whoosh?style=forceful&direction=fall&duration=1.8&seed=43",
            ],
        },
    },
    "rain": {
        "description": "A finite rain bed made from diffuse filtered noise and seeded surface impacts.",
        "parameters": {
            "intensity": {
                "type": "enum",
                "values": ["light", "steady", "heavy"],
                "default": "steady",
                "description": "Rain density and broadband wash level.",
            },
            "surface": {
                "type": "enum",
                "values": ["ground", "roof", "window"],
                "default": "ground",
                "description": "Surface character of the individual rain impacts.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 1.0,
                "maximum": 30.0,
                "default": 8.0,
                "description": "Length of the finite rain recording.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "rain falls steadily on a nearby surface",
                "a finite rain bed with distinct droplets is needed",
            ],
            "avoid_when": ["thunder is the main event", "a single close water drip is required"],
            "examples": [
                "sfx:rain?intensity=steady&surface=ground&duration=8.0&seed=42",
                "sfx:rain?intensity=heavy&surface=roof&duration=10.0&seed=43",
            ],
        },
    },
    "fire.crackle": {
        "description": "A finite low fire roar with sparse or active sharp crackling pops.",
        "parameters": {
            "activity": {
                "type": "enum",
                "values": ["quiet", "active"],
                "default": "quiet",
                "description": "Density of the crackling impulses.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 2.0,
                "maximum": 30.0,
                "default": 8.0,
                "description": "Length of the finite fire recording.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a small fire burns with an audible low roar and crackles",
                "a fireplace needs a finite background texture",
            ],
            "avoid_when": [
                "an explosion or large wildfire is intended",
                "a single isolated wood snap is required",
            ],
            "examples": [
                "sfx:fire.crackle?activity=quiet&duration=8.0&seed=42",
                "sfx:fire.crackle?activity=active&duration=12.0&seed=43",
            ],
        },
    },
    "birds.ambience": {
        "description": "A finite soft outdoor air bed with seeded short bird-call chirps.",
        "parameters": {
            "activity": {
                "type": "enum",
                "values": ["sparse", "busy"],
                "default": "sparse",
                "description": "Density of bird-call events.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 2.0,
                "maximum": 30.0,
                "default": 12.0,
                "description": "Length of the finite bird ambience.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "distant short bird calls punctuate an outdoor background",
                "a light natural bird ambience is needed",
            ],
            "avoid_when": [
                "a specific species call or close solo bird is required",
                "crickets or continuous wind is the focus",
            ],
            "examples": [
                "sfx:birds.ambience?activity=sparse&duration=12.0&seed=42",
                "sfx:birds.ambience?activity=busy&duration=16.0&seed=43",
            ],
        },
    },
    "crickets.ambience": {
        "description": "A finite high-frequency insect bed with seeded rhythmic cricket chirrup groups.",
        "parameters": {
            "activity": {
                "type": "enum",
                "values": ["sparse", "busy"],
                "default": "sparse",
                "description": "Density of cricket chirrup groups.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 2.0,
                "maximum": 30.0,
                "default": 12.0,
                "description": "Length of the finite cricket ambience.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "crickets chirr repeatedly in a nighttime outdoor scene",
                "a high rhythmic insect ambience is required",
            ],
            "avoid_when": [
                "bird calls or speech are the focus",
                "one close isolated insect is required",
            ],
            "examples": [
                "sfx:crickets.ambience?activity=sparse&duration=12.0&seed=42",
                "sfx:crickets.ambience?activity=busy&duration=16.0&seed=43",
            ],
        },
    },
    "keys.jingle": {
        "description": "A short keyring shake with stochastic metal micro-collisions and several tuned key resonances.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["light", "full"],
                "default": "light",
                "description": "Number and density of keys on the ring.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.25,
                "maximum": 5.0,
                "default": 1.2,
                "description": "Length of the keyring motion.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a small set of metal keys jingles",
                "keys are shaken or handled close to the listener",
            ],
            "avoid_when": [
                "a lock mechanism turns",
                "a large bell or other heavy metal object rings",
            ],
            "examples": [
                "sfx:keys.jingle?style=light&duration=1.2&seed=42",
                "sfx:keys.jingle?style=full&duration=2.0&seed=43",
            ],
        },
    },
    "lock.turn": {
        "description": "A key or deadbolt rotates through rough hardware friction and ends with a detent click.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["key", "deadbolt"],
                "default": "key",
                "description": "Small key-cylinder or heavier deadbolt hardware.",
            },
            "force": {
                "type": "enum",
                "values": ["light", "firm"],
                "default": "light",
                "description": "Effort of the rotation and terminal detent.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.3,
                "maximum": 5.0,
                "default": 1.5,
                "description": "Length of the lock-turn action.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": ["a key turns in a lock", "a deadbolt rotates and reaches its stop"],
            "avoid_when": ["keys jingle without engaging hardware", "a full door opens or closes"],
            "examples": [
                "sfx:lock.turn?style=key&force=light&duration=1.5&seed=42",
                "sfx:lock.turn?style=deadbolt&force=firm&duration=1.1&seed=43",
            ],
        },
    },
    "chair.move": {
        "description": "A loaded chair scrapes across a floor with stick-slip and structural floor response.",
        "parameters": {
            "surface": {
                "type": "enum",
                "values": ["wood", "stone", "carpet"],
                "default": "wood",
                "description": "Floor surface under the moving chair.",
            },
            "effort": {
                "type": "enum",
                "values": ["light", "firm"],
                "default": "light",
                "description": "Force applied while moving the chair.",
            },
            "action": {
                "type": "enum",
                "values": ["slide", "set_down"],
                "default": "slide",
                "description": "Whether the chair only slides or ends with a set-down contact.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.5,
                "maximum": 8.0,
                "default": 2.0,
                "description": "Length of the chair movement.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a chair is dragged across a floor",
                "a chair shifts with a scrape and light floor creak",
            ],
            "avoid_when": [
                "a person walks without moving furniture",
                "a close door or drawer slides",
            ],
            "examples": [
                "sfx:chair.move?surface=wood&effort=light&action=slide&duration=2.0&seed=42",
                "sfx:chair.move?surface=stone&effort=firm&action=set_down&duration=2.8&seed=43",
            ],
        },
    },
    "cloth.rustle": {
        "description": "A fabric swish with soft broadband friction, irregular flutter, and short cloth resonances.",
        "parameters": {
            "fabric": {
                "type": "enum",
                "values": ["cotton", "silk", "nylon"],
                "default": "cotton",
                "description": "Broad material character of the fabric.",
            },
            "activity": {
                "type": "enum",
                "values": ["light", "active"],
                "default": "light",
                "description": "Density and strength of the fabric motion.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.5,
                "maximum": 12.0,
                "default": 4.0,
                "description": "Length of the cloth movement.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "clothing or loose fabric rustles during movement",
                "a short fabric swish is audible",
            ],
            "avoid_when": ["paper is handled or turned", "a loud plastic wrapper crinkles"],
            "examples": [
                "sfx:cloth.rustle?fabric=cotton&activity=light&duration=4.0&seed=42",
                "sfx:cloth.rustle?fabric=silk&activity=active&duration=3.0&seed=43",
            ],
        },
    },
    "floor.creak": {
        "description": "A loaded floorboard flexes with low frictional creak and structural resonance.",
        "parameters": {
            "surface": {
                "type": "enum",
                "values": ["wood", "carpet"],
                "default": "wood",
                "description": "Wooden floor or carpeted subfloor response.",
            },
            "weight": {
                "type": "enum",
                "values": ["light", "heavy"],
                "default": "light",
                "description": "Load placed on the floor during the shift.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.4,
                "maximum": 4.0,
                "default": 1.3,
                "description": "Length of the weight-shift creak.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a floorboard creaks under shifting weight",
                "a step flexes a wooden floor",
            ],
            "avoid_when": [
                "footstep impacts alone are needed",
                "a chair is dragged across the floor",
            ],
            "examples": [
                "sfx:floor.creak?surface=wood&weight=light&duration=1.3&seed=42",
                "sfx:floor.creak?surface=carpet&weight=heavy&duration=1.8&seed=43",
            ],
        },
    },
    "keyboard.typing": {
        "description": "Naturalistic procedural computer-keyboard typing with varied key presses, release/top-out sounds, larger-key events, and humanized timing.",
        "parameters": {
            "speed": {
                "type": "enum",
                "values": ["slow", "steady", "fast"],
                "default": "steady",
                "description": "Rate and density of keystrokes.",
            },
            "force": {
                "type": "enum",
                "values": ["light", "firm"],
                "default": "light",
                "description": "Strength of each key press.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 1.0,
                "maximum": 30.0,
                "default": 6.0,
                "description": "Length of the typing sequence.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a keyboard is typed at a slow, steady, or fast pace",
                "distant office typing is needed without speech",
            ],
            "avoid_when": [
                "a single button is pressed",
                "a close printer or typewriter action is required",
            ],
            "examples": [
                "sfx:keyboard.typing?speed=steady&force=light&duration=6.0&seed=42",
                "sfx:keyboard.typing?speed=fast&force=firm&duration=8.0&seed=43",
            ],
        },
    },
    "clock.tick": {
        "description": "A regular escapement tick-tock excites the clock case at a steady rate.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["wall", "mantel"],
                "default": "wall",
                "description": "Lighter wall-clock or heavier mantel-clock body.",
            },
            "rate": {
                "type": "enum",
                "values": ["slow", "normal"],
                "default": "normal",
                "description": "One tick per second or a quicker tick-tock beat.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 1.0,
                "maximum": 30.0,
                "default": 6.0,
                "description": "Length of the ticking sequence.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a clock ticks regularly in the room",
                "a short mechanical escapement tick-tock is heard",
            ],
            "avoid_when": ["an alarm rings", "a digital clock emits a beep"],
            "examples": [
                "sfx:clock.tick?style=wall&rate=normal&duration=6.0&seed=42",
                "sfx:clock.tick?style=mantel&rate=slow&duration=8.0&seed=43",
            ],
        },
    },
    "alarm.ring": {
        "description": "A mechanical bell or electronic alarm rings continuously or in spaced bursts.",
        "parameters": {
            "style": {
                "type": "enum",
                "values": ["mechanical", "electronic"],
                "default": "mechanical",
                "description": "Struck twin-bell or electronic alarm sound.",
            },
            "pattern": {
                "type": "enum",
                "values": ["intermittent", "continuous"],
                "default": "intermittent",
                "description": "Whether the alarm pauses between ringing bursts.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 1.0,
                "maximum": 15.0,
                "default": 5.0,
                "description": "Length of the finite alarm recording.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a mechanical or electronic alarm rings",
                "an alarm repeats in short bursts",
            ],
            "avoid_when": [
                "a phone receives a quiet notification",
                "a clock ticks without ringing",
            ],
            "examples": [
                "sfx:alarm.ring?style=mechanical&pattern=intermittent&duration=5.0&seed=42",
                "sfx:alarm.ring?style=electronic&pattern=continuous&duration=4.0&seed=43",
            ],
        },
    },
    "elevator.arrive": {
        "description": "An elevator motor slows, gives a floor chime, and slides its doors into position.",
        "parameters": {
            "size": {
                "type": "enum",
                "values": ["small", "large"],
                "default": "small",
                "description": "Size and motor character of the elevator.",
            },
            "chime": {
                "type": "enum",
                "values": ["single", "double"],
                "default": "single",
                "description": "Number of short arrival chimes.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 1.0,
                "maximum": 8.0,
                "default": 3.5,
                "description": "Length of the arrival and door action.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "an elevator arrives at a floor with a chime and sliding doors",
                "a lift slows and opens its doors",
            ],
            "avoid_when": [
                "a standalone electronic notification",
                "a whole elevator ride is required",
            ],
            "examples": [
                "sfx:elevator.arrive?size=small&chime=single&duration=3.5&seed=42",
                "sfx:elevator.arrive?size=large&chime=double&duration=4.0&seed=43",
            ],
        },
    },
    "water.pour": {
        "description": "A finite turbulent pour with splash events and optional vessel resonance.",
        "parameters": {
            "flow": {
                "type": "enum",
                "values": ["trickle", "steady", "strong"],
                "default": "steady",
                "description": "Rate and weight of the water flow.",
            },
            "vessel": {
                "type": "enum",
                "values": ["glass", "ceramic", "metal"],
                "default": "glass",
                "description": "Container resonance excited by the pour.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.5,
                "maximum": 12.0,
                "default": 4.0,
                "description": "Length of the finite pour.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "water is poured into a vessel",
                "a short liquid stream ends with light splashes",
            ],
            "avoid_when": ["a faucet runs continuously", "a single drip or splash is the focus"],
            "examples": [
                "sfx:water.pour?flow=steady&vessel=glass&duration=4.0&seed=42",
                "sfx:water.pour?flow=strong&vessel=metal&duration=3.0&seed=43",
            ],
        },
    },
    "water.running": {
        "description": "A finite continuous turbulent stream with gently changing flow and light splashes.",
        "parameters": {
            "flow": {
                "type": "enum",
                "values": ["gentle", "steady", "strong"],
                "default": "steady",
                "description": "Strength of the running water stream.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 1.0,
                "maximum": 20.0,
                "default": 6.0,
                "description": "Length of the finite water stream.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a faucet or stream runs continuously",
                "steady water flow is needed beneath another action",
            ],
            "avoid_when": [
                "water is poured into a container",
                "a single isolated drip is required",
            ],
            "examples": [
                "sfx:water.running?flow=steady&duration=6.0&seed=42",
                "sfx:water.running?flow=strong&duration=8.0&seed=43",
            ],
        },
    },
    "crowd.murmur": {
        "description": "A non-lexical crowd texture made from overlapping filtered noise grains, with no intelligible speech.",
        "parameters": {
            "density": {
                "type": "enum",
                "values": ["sparse", "moderate", "busy"],
                "default": "moderate",
                "description": "Density of the overlapping crowd texture.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 2.0,
                "maximum": 30.0,
                "default": 8.0,
                "description": "Length of the finite crowd ambience.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a distant crowd murmurs indistinctly",
                "a nonverbal crowd bed is needed under narration",
            ],
            "avoid_when": [
                "intelligible dialogue or words are required",
                "a single person speaks close to the listener",
            ],
            "examples": [
                "sfx:crowd.murmur?density=moderate&duration=8.0&seed=42",
                "sfx:crowd.murmur?density=busy&duration=12.0&seed=43",
            ],
        },
    },
    "thunder": {
        "description": "A finite distributed low-frequency thunder roll with a short crack and decaying rumble.",
        "parameters": {
            "intensity": {
                "type": "enum",
                "values": ["light", "strong"],
                "default": "strong",
                "description": "Overall energy of the thunder event.",
            },
            "distance": {
                "type": "enum",
                "values": ["near", "distant"],
                "default": "distant",
                "description": "Spectral weight and attack character of the rumble.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 3.0,
                "maximum": 15.0,
                "default": 6.0,
                "description": "Length of the thunder roll and decay.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a single thunder roll follows or accompanies a storm",
                "low distant thunder rumbles and decays",
            ],
            "avoid_when": [
                "continuous rain is the only required sound",
                "an explosion or designed impact is intended",
            ],
            "examples": [
                "sfx:thunder?intensity=strong&distance=distant&duration=6.0&seed=42",
                "sfx:thunder?intensity=light&distance=near&duration=5.0&seed=43",
            ],
        },
    },
    "car.door": {
        "description": "A car door hinge moves against its body cavity and ends with a firm or light latch contact.",
        "parameters": {
            "action": {
                "type": "enum",
                "values": ["open", "close"],
                "default": "close",
                "description": "Whether the door opens or closes.",
            },
            "size": {
                "type": "enum",
                "values": ["sedan", "suv"],
                "default": "sedan",
                "description": "Size and low-frequency body response of the car door.",
            },
            "force": {
                "type": "enum",
                "values": ["light", "firm"],
                "default": "firm",
                "description": "Force of the hinge motion and latch contact.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 0.5,
                "maximum": 4.0,
                "default": 1.5,
                "description": "Length of the car-door action.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a car door opens or shuts with a body-panel clunk",
                "a vehicle door latch closes",
            ],
            "avoid_when": ["an indoor room door opens or closes", "a car engine starts"],
            "examples": [
                "sfx:car.door?action=close&size=sedan&force=firm&duration=1.5&seed=42",
                "sfx:car.door?action=open&size=suv&force=light&duration=1.8&seed=43",
            ],
        },
    },
    "car.engine": {
        "description": "A finite rotating combustion-engine source that starts, idles, or revs with exhaust texture.",
        "parameters": {
            "action": {
                "type": "enum",
                "values": ["start", "idle", "rev"],
                "default": "idle",
                "description": "Engine action for this finite recording.",
            },
            "vehicle": {
                "type": "enum",
                "values": ["compact", "truck"],
                "default": "compact",
                "description": "Engine size and harmonic weight.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 1.0,
                "maximum": 15.0,
                "default": 5.0,
                "description": "Length of the engine recording.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a car engine starts, idles, or revs in place",
                "a nearby vehicle motor runs without passing",
            ],
            "avoid_when": ["a vehicle drives past the listener", "a generic appliance motor runs"],
            "examples": [
                "sfx:car.engine?action=idle&vehicle=compact&duration=5.0&seed=42",
                "sfx:car.engine?action=rev&vehicle=truck&duration=4.0&seed=43",
            ],
        },
    },
    "car.passby": {
        "description": "A finite vehicle pass with a moving engine-speed curve and road turbulence in mono.",
        "parameters": {
            "speed": {
                "type": "enum",
                "values": ["slow", "fast"],
                "default": "slow",
                "description": "Vehicle speed and pitch movement through the pass.",
            },
            "vehicle": {
                "type": "enum",
                "values": ["compact", "truck"],
                "default": "compact",
                "description": "Vehicle engine and road-noise weight.",
            },
            "duration": {
                "type": "seconds",
                "minimum": 2.0,
                "maximum": 10.0,
                "default": 5.0,
                "description": "Length of the finite vehicle pass.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative deterministic variation seed.",
            },
        },
        "llm": {
            "use_when": [
                "a vehicle passes with a clear rise and fall in engine pitch and level",
                "a mono road pass-by needs engine and tire wash",
            ],
            "avoid_when": [
                "a stationary engine idles or revs",
                "stereo direction or a specific perspective is required",
            ],
            "examples": [
                "sfx:car.passby?speed=slow&vehicle=compact&duration=5.0&seed=42",
                "sfx:car.passby?speed=fast&vehicle=truck&duration=4.0&seed=43",
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
