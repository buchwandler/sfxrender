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

Built-in footsteps and knocks use layered, deterministic procedural synthesis rather than bundled
recordings: footsteps combine heel/body, surface response, and delayed sole contacts, while knocks
combine physical contact, a panel/body response, inharmonic modes, and colored diffusion. Walking
uses subtle seeded timing variation; repeated renders with the same URI remain reproducible.

Built-in renderers ignore unknown query parameters so callers can preserve forward-compatible URI fields; misspelled parameters are therefore not rejected.

## Audition the catalog

Install the project in editable mode and render the fixed-seed listening catalog:

```bash
python -m pip install -e .
python examples/render_catalog.py
```

Individual WAVs and `00-catalog-showcase.wav` are written to `example-artifacts/`. Generated WAV and JSON files there are intentionally gitignored; the tracked `.gitkeep` preserves the output directory.

Run the focused Foley comparison gallery with:

```bash
python examples/foley_quality_gallery.py
```

It writes six footsteps, four knocks, and a combined `foley-quality-showcase.wav` to
`example-artifacts/`; the showcase leaves 550 ms between cases. Third-party recordings used
for local listening or measurement belong under the gitignored `reference-audio/` directory.
They are not fetched, committed, included in tests, or packaged.

## Examples

- [`examples/render_catalog.py`](examples/render_catalog.py) — individual catalog renders and a combined showcase.
- [`examples/audiobook_scene.py`](examples/audiobook_scene.py) — a short composed arrival scene with footsteps, knocks, and a door opening.
- [`examples/parameter_gallery.py`](examples/parameter_gallery.py) — same-seed comparisons of knock materials and walking surfaces.
- [`examples/foley_quality_gallery.py`](examples/foley_quality_gallery.py) — fixed-seed Foley surface, footwear, and knock comparisons.

Run the other examples after installation:

```bash
python examples/audiobook_scene.py
python examples/parameter_gallery.py
python examples/foley_quality_gallery.py
```

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
