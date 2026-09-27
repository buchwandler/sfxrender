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

The door actions preserve a stable generated object identity across actions when their material and seed match:

```bash
sfxrender 'sfx:door.open?material=wood&speed=slow&creak=0.75&seed=31' -o open.wav
sfxrender 'sfx:door.close?material=wood&speed=fast&force=0.85&creak=0.2&seed=31' -o close.wav
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
- `door.close`

Built-in footsteps and knocks use deterministic, NumPy-only procedural synthesis rather than bundled recordings. Footsteps build a seeded multi-stage ground-reaction-force envelope; footwear-shaped broadband contact excites either stable wood/stone/carpet modes or GRF-driven stochastic gravel impacts. Low-level seeded friction and release texture preserve event variation without a pitched body oscillator. Knocks use the same broadband impact and damped modal-response primitives. Walking retains subtle seeded timing and side variation, so repeated renders with the same URI match exactly.

Door effects use seeded physically informed models. A generated door has stable panel/frame resonances, hinge-friction regions, and hardware character; opening and closing actions excite that same model differently. The same material and seed intentionally represent the same door for both actions.

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

It writes six ordered footstep WAVs (`01-footsteps-wood-boots.wav` through `06-footsteps-gravel-boots.wav`), four knocks, a footstep-only `foley-footsteps-showcase.wav`, and a combined `foley-quality-showcase.wav` under `example-artifacts/`; both strips leave 550 ms between cases. Third-party recordings used for local listening or measurement belong under the gitignored `reference-audio/` directory. They are not fetched, bundled, packaged, or committed.

Render the door-specific fixed-seed gallery with:

```bash
python examples/door_quality_gallery.py
```

It writes eight ordered open/close WAVs and `door-quality-showcase.wav` under `example-artifacts/`; the wood door #501 cases pair opening and closing from the same generated identity.

Run the deterministic spectral/temporal analysis report with:

```bash
python examples/analyze_foley.py
```

It writes `foley-analysis.txt` under `example-artifacts/` (or accepts a custom output directory when called from Python).

Use the NumPy-only door waveform/model metrics report with:

```bash
python examples/analyze_doors.py
```

It writes `door-analysis.txt` beneath `example-artifacts/` (or a caller-supplied output directory).

## Examples

- [`examples/render_catalog.py`](examples/render_catalog.py) — individual catalog renders and a combined showcase.
- [`examples/audiobook_scene.py`](examples/audiobook_scene.py) — a short composed arrival scene with footsteps, knocks, and a door opening.
- [`examples/parameter_gallery.py`](examples/parameter_gallery.py) — same-seed comparisons of knock materials and walking surfaces.
- [`examples/foley_quality_gallery.py`](examples/foley_quality_gallery.py) — fixed-seed Foley surface, footwear, and knock comparisons.
- [`examples/analyze_foley.py`](examples/analyze_foley.py) — reproducible spectral and contact-stage metrics for the fixed footstep cases.
- [`examples/door_quality_gallery.py`](examples/door_quality_gallery.py) — fixed-seed opening/closing gallery, including a same-seed paired door.
- [`examples/analyze_doors.py`](examples/analyze_doors.py) — NumPy-only waveform and generated door-model metrics.

Run the other examples after installation:

```bash
python examples/audiobook_scene.py
python examples/parameter_gallery.py
python examples/foley_quality_gallery.py
python examples/analyze_foley.py
python examples/door_quality_gallery.py
python examples/analyze_doors.py
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
