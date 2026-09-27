import numpy as np
import pytest

from sfxrender import RenderContext, RenderedSound, SFXRenderer, SFXRenderError, SfxSpec


def test_custom_renderer() -> None:
    renderer = SFXRenderer(sample_rate=8_000)

    def silence(spec: SfxSpec, context: RenderContext) -> RenderedSound:
        return RenderedSound(np.zeros(80, dtype=np.float32), context.sample_rate, spec)

    renderer.register("custom.silence", silence)
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
