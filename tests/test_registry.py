import numpy as np
import pytest

from sfxrender import (
    InvalidEffectParameterError,
    RenderContext,
    RenderedSound,
    SFXRenderer,
    SFXRenderError,
    SfxSpec,
)


def test_custom_renderer() -> None:
    renderer = SFXRenderer(sample_rate=8_000)

    def silence(spec: SfxSpec, context: RenderContext) -> RenderedSound:
        return RenderedSound(np.zeros(80, dtype=np.float32), context.sample_rate, spec)

    renderer.register("custom.silence", silence)
    assert "custom.silence" in renderer.effects()
    assert "custom.silence" not in renderer.catalog()
    assert "custom.silence" not in renderer.llm_catalog()["effects"]
    spec = renderer.validate_uri("sfx:custom.silence?free_form=renderer-defined")
    assert spec.parameters == {"free_form": "renderer-defined"}
    sound = renderer.render_uri("sfx:custom.silence")
    assert sound.samples.size == 80


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((1, 80), dtype=np.float32),
        np.array([1.1], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.zeros(80, dtype=np.float64),
        np.zeros(0, dtype=np.float32),
    ],
    ids=["multichannel", "out-of-range", "non-finite", "wrong-dtype", "empty"],
)
def test_custom_renderer_pcm_must_obey_public_contract(samples: np.ndarray) -> None:
    renderer = SFXRenderer(sample_rate=8_000)
    renderer.register(
        "custom.invalid",
        lambda spec, context: RenderedSound(samples, context.sample_rate, spec),
    )
    with pytest.raises(SFXRenderError):
        renderer.render_uri("sfx:custom.invalid")


def test_duplicate_registration_requires_replace() -> None:
    renderer = SFXRenderer()
    with pytest.raises(ValueError):
        renderer.register("impact.knock", lambda spec, _context: renderer.render(spec))


def _custom_descriptor(description: str = "A documented keyboard sound.") -> dict[str, object]:
    return {
        "description": description,
        "parameters": {},
        "llm": {
            "use_when": ["someone types on a keyboard"],
            "avoid_when": [],
            "examples": ["sfx:custom.keyboard"],
        },
    }


def test_described_custom_renderer_is_runtime_catalog_discoverable() -> None:
    renderer = SFXRenderer()

    def callback(spec: SfxSpec, _context: RenderContext) -> RenderedSound:
        return renderer.render(spec)

    renderer.register("custom.keyboard", callback, descriptor=_custom_descriptor())

    assert "custom.keyboard" in renderer.effects()
    assert renderer.catalog()["custom.keyboard"] == _custom_descriptor()
    assert "custom.keyboard" in renderer.llm_catalog()["effects"]

    exposed = renderer.catalog()
    exposed["custom.keyboard"]["llm"]["examples"].clear()
    assert renderer.catalog()["custom.keyboard"]["llm"]["examples"]


def test_replacement_preserves_or_atomically_updates_descriptors() -> None:
    renderer = SFXRenderer()

    def original(spec: SfxSpec, _context: RenderContext) -> RenderedSound:
        return renderer.render(spec)

    renderer.register("custom.keyboard", original, descriptor=_custom_descriptor())
    original_descriptor = renderer.catalog()["custom.keyboard"]

    def replacement(spec: SfxSpec, _context: RenderContext) -> RenderedSound:
        return renderer.render(spec)

    renderer.register("custom.keyboard", replacement, replace=True)
    assert renderer._renderers["custom.keyboard"] is replacement
    assert renderer.catalog()["custom.keyboard"] == original_descriptor

    updated_descriptor = _custom_descriptor("A different documented keyboard sound.")

    def updated_renderer(spec: SfxSpec, _context: RenderContext) -> RenderedSound:
        return renderer.render(spec)

    renderer.register(
        "custom.keyboard",
        updated_renderer,
        descriptor=updated_descriptor,
        replace=True,
    )
    assert renderer._renderers["custom.keyboard"] is updated_renderer
    assert renderer.catalog()["custom.keyboard"] == updated_descriptor

    invalid_descriptor = {"description": "bad", "parameters": "not a mapping"}
    with pytest.raises(ValueError):
        renderer.register(
            "custom.keyboard",
            original,
            descriptor=invalid_descriptor,
            replace=True,
        )
    assert renderer._renderers["custom.keyboard"] is updated_renderer
    assert renderer.catalog()["custom.keyboard"] == updated_descriptor


def test_invalid_new_descriptor_does_not_register_renderer() -> None:
    renderer = SFXRenderer()

    def callback(spec: SfxSpec, _context: RenderContext) -> RenderedSound:
        return renderer.render(spec)

    with pytest.raises(ValueError):
        renderer.register(
            "custom.invalid",
            callback,
            descriptor={"description": "invalid", "parameters": None},
        )
    assert "custom.invalid" not in renderer.effects()
    assert "custom.invalid" not in renderer.catalog()


def _custom_parameter_descriptor() -> dict[str, object]:
    return {
        "description": "A custom keyboard typing effect.",
        "parameters": {
            "duration": {
                "type": "seconds",
                "minimum": 0.2,
                "maximum": 30.0,
                "default": 2.0,
                "description": "How long typing continues.",
            },
            "speed": {
                "type": "enum",
                "values": ["slow", "normal", "fast"],
                "default": "normal",
                "description": "Typing cadence.",
            },
            "ticket": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "required": True,
                "description": "Number of typing bursts.",
            },
            "seed": {
                "type": "integer",
                "required": False,
                "description": "Optional non-negative variation seed.",
            },
        },
        "llm": {
            "use_when": ["someone types audibly on a keyboard"],
            "avoid_when": [],
            "examples": ["sfx:custom.keyboard?duration=2&speed=normal&ticket=2&seed=42"],
        },
    }


def test_described_custom_effects_use_descriptor_validation() -> None:
    renderer = SFXRenderer()

    def callback(spec: SfxSpec, _context: RenderContext) -> RenderedSound:
        return renderer.render(spec)

    renderer.register(
        "custom.keyboard",
        callback,
        descriptor=_custom_parameter_descriptor(),
    )

    spec = renderer.validate_uri("sfx:custom.keyboard?duration=2.5&speed=fast&ticket=3&seed=41")
    assert spec.effect == "custom.keyboard"
    assert spec.parameters["ticket"] == "3"

    with pytest.raises(InvalidEffectParameterError, match="0.2..30.0"):
        renderer.validate_uri("sfx:custom.keyboard?duration=31&speed=fast&ticket=3")
    with pytest.raises(InvalidEffectParameterError, match="expected one of"):
        renderer.validate_uri("sfx:custom.keyboard?duration=2&speed=instant&ticket=3")
    with pytest.raises(InvalidEffectParameterError, match="parameter is required"):
        renderer.validate_uri("sfx:custom.keyboard?duration=2&speed=normal")
    with pytest.raises(InvalidEffectParameterError, match="unknown parameter"):
        renderer.validate_uri("sfx:custom.keyboard?duration=2&speed=normal&ticket=3&unknown=1")
