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

## Using SFXRender as an application media resolver

SFXRender provides `SFXRenderer.render_uri()` as the stable entry point for consuming applications such as Readio. Construct it with the source rendering rate, pass the complete `sfx:` URI, and consume the returned mono `float32` PCM:

```python
from sfxrender import SFXRenderer

renderer = SFXRenderer(sample_rate=24_000)

result = renderer.render_uri(
    "sfx:impact.knock?material=oak&count=3&force=0.7&seed=42"
)
pcm = result.samples
```

The result also provides `sample_rate`, `duration`, and the parsed `spec`; `sfxrender.__version__` is available for renderer/package cache provenance. URI, effect, and parameter failures use public SFXRender exceptions so applications can map them to stable diagnostics.

An SSMD author can use the same source URI in a `src` attribute:

```ssmd
[three knocks]{src="sfx:impact.knock?material=oak&count=3&force=0.7&seed=42"}
```

SFXRender does **not** parse SSMD. The consuming application parses its document and resolves the URI by calling SFXRender; SFXRender returns PCM. The complete `sfx:` URI is the cross-library integration boundary and procedural-source identity. The application remains responsible for dispatching other sources such as files or HTTP URLs.

## Built-in MVP effects

- `impact.knock`
- `footsteps.walk`
- `phone.ring`
- `door.open`
- `door.close`
- `printer.print`
- `printer.tray_open`
- `printer.power_switch`
- `printer.restart`
- `pen.write`
- `printer.wake`

Built-in footsteps and knocks use deterministic, NumPy-only procedural synthesis rather than bundled recordings. Footsteps retain the staged ground-reaction-force and heel/sole/toe timing: compliant contact excites effective floor modes, footwear/floor presets shape contact and radiation, and bounded friction/roughness and gravel events draw energy from load and slip. Knocks use a compliant force pulse, geometry-aware modal objects, and strike-position coupling rather than random body-frequency jitter. Fixed URIs render finite mono float audio deterministically, including seeded step-to-step variation.

Door effects use a shared effective assembly with stable panel/frame resonances, inertia-loaded hinge friction, spatial roughness, and physical latch/stop contacts. Open and close actions excite the same generated door differently; matching material and seed intentionally identify the same object. Semantic controls such as `speed`, `creak`, and `force` alter its motion/contact behavior without changing the public URI/API.

Built-in renderers validate parameter names and values against SFXRender's effect catalog; unknown names and invalid values raise public typed errors instead of being silently ignored.

## Printer-story audio spans

SFXRender provides deterministic printer and pen events through semantic URIs; the consuming application resolves `src` rather than having SSMD synthesize audio. For example, the printer story can include:

```ssmd
[printer prints three sheets]{src="sfx:printer.print?pages=3&speed=normal&seed=301"}

[paper tray opens]{src="sfx:printer.tray_open?paper_load=full&seed=302"}

[pen writes on paper]{src="sfx:pen.write?duration=1.8&pressure=0.55&seed=305"}
```

When audio is supported, the bracket text is descriptive audio metadata, not spoken narration. These spans are foreground audio events; overlap/mixing with narration is the consuming application's responsibility.

## Effective models and reference calibration

These are physically informed **effective models**, not full material identification or a claim of exact boundary conditions. Modal frequencies, gains, contact parameters, and damping are practical acoustic approximations; the generated audio does not model the recording room or microphone. Use several similarly recorded hits, and treat presets as empirical tuning data. No mesh/FEM solver or SciPy dependency is used by rendering or the calibration tools.

The developer-only NumPy tools accept uncompressed integer PCM WAVs. Fit a modal preset from one or more takes, then compare rendered output against reference takes:

```bash
python tools/fit_modal_reference.py --name oak --output example-artifacts/oak-modes.json reference-a.wav reference-b.wav
python tools/compare_reference.py --reference reference-a.wav reference-b.wav --candidate render-a.wav render-b.wav --output example-artifacts/oak-comparison.json
```

The fitter detects an RMS onset, analyzes the configurable decay window (default 100–500 ms), estimates interpolated spectral peaks and narrowband log-amplitude decay constants, and clusters repeatable modes. Its JSON includes modal gain estimates, support, and frequency/decay spread. The comparison report does not align waveforms: it summarizes feature medians/spread, band energy, centroid trajectory, event density, matched modal shifts, and inter-hit spectral correlation. These measurements are sensitive to background noise, room response, microphone placement, and short/noisy decays; they are calibration aids, not ground truth. See `python tools/fit_modal_reference.py --help` and `python tools/compare_reference.py --help` for options.

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

It writes eight ordered footstep WAVs, including same-seed footwear and light/heavy-load A/B cases, six knocks including same-seed wood/oak and soft-fingertip/hard-metal-impactor contrasts, a footstep-only `foley-footsteps-showcase.wav`, and a combined `foley-quality-showcase.wav` under `example-artifacts/`; both strips leave 550 ms between cases. Third-party recordings for local listening or measurement belong under the gitignored `reference-audio/` directory. They are not fetched, bundled, packaged, or committed.

Render the door-specific fixed-seed gallery with:

```bash
python examples/door_quality_gallery.py
```

It writes ten ordered open/close WAVs and `door-quality-showcase.wav` under `example-artifacts/`. Wood cases share seed 501 for controlled speed, creak, and closing-force contrasts; opening and closing from that seed reuse the same generated door.

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
- [`tools/fit_modal_reference.py`](tools/fit_modal_reference.py) and [`tools/compare_reference.py`](tools/compare_reference.py) — NumPy-only developer calibration tools; see [Effective models and reference calibration](#effective-models-and-reference-calibration).

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
