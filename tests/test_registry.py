import numpy as np
import pytest

from sfxrender import RenderContext, RenderedSound, SFXRenderer, SfxSpec


def test_custom_renderer() -> None:
    renderer = SFXRenderer(sample_rate=8_000)

    def silence(spec: SfxSpec, context: RenderContext) -> RenderedSound:
        return RenderedSound(np.zeros(80, dtype=np.float32), context.sample_rate, spec)

    renderer.register("custom.silence", silence)
    sound = renderer.render_uri("sfx:custom.silence")
    assert sound.samples.size == 80


def test_duplicate_registration_requires_replace() -> None:
    renderer = SFXRenderer()
    with pytest.raises(ValueError):
        renderer.register("impact.knock", lambda spec, context: renderer.render(spec))
