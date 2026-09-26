# sfxrender

`SFXRender` is a small deterministic sound-effect renderer intended for audiobook and
SSMD/Readio workflows.

The MVP intentionally does **not** depend on a generative model. It accepts semantic
`sfx:` URIs and turns a small built-in vocabulary into NumPy PCM audio. Applications can
register additional renderers later for sample-based, ONNX, API, or other backends.

## Install

```bash
python -m pip install -e .
```

Versioning is dynamic via Git tags and `setuptools-scm`. A source tree without Git metadata
uses `0+unknown` so exported ZIPs remain buildable.

## Example

```python
from sfxrender import SFXRenderer

renderer = SFXRenderer(sample_rate=24_000)

sound = renderer.render_uri(
    "sfx:impact.knock?material=oak&count=3&force=0.7&seed=42"
)
sound.write_wav("knock.wav")
```

Or from the CLI:

```bash
sfxrender 'sfx:footsteps.walk?surface=wood&footwear=boots&count=5&seed=42' -o steps.wav
```

## SSMD-facing form

SFXRender is designed around URIs that can live in SSMD `src` attributes:

```ssmd
[three knocks]{src="sfx:impact.knock?material=oak&count=3&force=0.7&seed=42"}
```

Readio can later resolve `sfx:` sources through this package while leaving ordinary file or
HTTP audio sources unchanged.

## Built-in MVP effects

- `impact.knock`
- `footsteps.walk`
- `phone.ring`
- `door.open`

The sound design is deliberately simple. The API and deterministic URI contract are the
important parts of the MVP.

## Extension

```python
from sfxrender import RenderContext, RenderedSound, SFXRenderer, SfxSpec

renderer = SFXRenderer()

def custom(spec: SfxSpec, context: RenderContext) -> RenderedSound:
    ...

renderer.register("custom.effect", custom)
```

## License

Apache-2.0
