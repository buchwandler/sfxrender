from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from sfxrender import catalog
from sfxrender.catalog import validate_effect_descriptor


def _descriptor() -> dict[str, Any]:
    return {
        "description": "A short test tone.",
        "parameters": {
            "mode": {
                "type": "enum",
                "values": ["quiet", "loud"],
                "default": "quiet",
                "description": "Tone level.",
            },
            "level": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
                "description": "Relative output level.",
            },
        },
        "llm": {
            "use_when": ["a test tone is audible"],
            "avoid_when": [],
            "examples": ["sfx:test.beep?mode=quiet&level=0.5"],
        },
    }


def test_all_builtin_descriptors_pass_descriptor_validation() -> None:
    for effect, descriptor in catalog().items():
        normalized = validate_effect_descriptor(effect, descriptor)
        assert normalized == descriptor


def test_descriptor_validation_returns_a_defensive_copy() -> None:
    descriptor = _descriptor()
    normalized = validate_effect_descriptor("test.beep", descriptor)
    normalized["parameters"]["mode"]["values"].clear()
    assert descriptor["parameters"]["mode"]["values"] == ["quiet", "loud"]


def test_non_llm_descriptor_may_omit_parameter_descriptions() -> None:
    descriptor = _descriptor()
    descriptor.pop("llm")
    descriptor["parameters"]["mode"].pop("description")
    validated = validate_effect_descriptor("test.beep", descriptor)
    assert "description" not in validated["parameters"]["mode"]


@pytest.mark.parametrize(
    "case",
    [
        "missing_description",
        "empty_description",
        "missing_parameters",
        "unsupported_parameter_type",
        "enum_without_values",
        "duplicate_enum_values",
        "minimum_exceeds_maximum",
        "default_outside_numeric_bounds",
        "default_outside_enum_values",
        "non_string_use_when",
        "empty_use_when",
        "example_targets_another_effect",
        "malformed_example_uri",
        "example_has_invalid_parameter",
        "llm_parameter_missing_description",
        "non_finite_bound",
        "non_boolean_required",
        "null_numeric_bound",
        "unhashable_parameter_type",
        "required_parameter_missing_from_example",
    ],
)
def test_malformed_descriptors_are_rejected(case: str) -> None:
    descriptor = deepcopy(_descriptor())
    if case == "missing_description":
        descriptor.pop("description")
    elif case == "empty_description":
        descriptor["description"] = "  "
    elif case == "missing_parameters":
        descriptor.pop("parameters")
    elif case == "unsupported_parameter_type":
        descriptor["parameters"]["level"]["type"] = "ratio"
    elif case == "enum_without_values":
        descriptor["parameters"]["mode"].pop("values")
    elif case == "duplicate_enum_values":
        descriptor["parameters"]["mode"]["values"] = ["quiet", "quiet"]
    elif case == "minimum_exceeds_maximum":
        descriptor["parameters"]["level"]["minimum"] = 2.0
    elif case == "default_outside_numeric_bounds":
        descriptor["parameters"]["level"]["default"] = 1.5
    elif case == "default_outside_enum_values":
        descriptor["parameters"]["mode"]["default"] = "medium"
    elif case == "non_string_use_when":
        descriptor["llm"]["use_when"] = [1]
    elif case == "empty_use_when":
        descriptor["llm"]["use_when"] = []
    elif case == "example_targets_another_effect":
        descriptor["llm"]["examples"] = ["sfx:other.effect?mode=quiet"]
    elif case == "malformed_example_uri":
        descriptor["llm"]["examples"] = ["not-an-sfx-uri"]
    elif case == "example_has_invalid_parameter":
        descriptor["llm"]["examples"] = ["sfx:test.beep?mode=quiet&level=2"]
    elif case == "llm_parameter_missing_description":
        descriptor["parameters"]["mode"].pop("description")
    elif case == "non_finite_bound":
        descriptor["parameters"]["level"]["maximum"] = float("inf")
    elif case == "non_boolean_required":
        descriptor["parameters"]["mode"]["required"] = "yes"
    elif case == "null_numeric_bound":
        descriptor["parameters"]["level"]["minimum"] = None
    elif case == "unhashable_parameter_type":
        descriptor["parameters"]["level"]["type"] = ["number"]
    elif case == "required_parameter_missing_from_example":
        descriptor["parameters"]["level"]["required"] = True
        descriptor["llm"]["examples"] = ["sfx:test.beep?mode=quiet"]

    with pytest.raises(ValueError):
        validate_effect_descriptor("test.beep", descriptor)
