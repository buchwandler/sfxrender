"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from . import __version__
from .catalog import catalog
from .core import SFXRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sfxrender", description="Render an sfx: URI to WAV")
    parser.add_argument("uri", nargs="?", help="e.g. sfx:impact.knock?material=oak&count=3")
    parser.add_argument("-o", "--output", default="sfx.wav", help="output WAV path")
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--list-effects", action="store_true")
    parser.add_argument(
        "--catalog", action="store_true", help="print built-in effect catalog as JSON"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--llm-catalog",
        action="store_true",
        help="print the LLM authoring capability catalog as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    renderer = SFXRenderer(sample_rate=args.sample_rate)
    if args.catalog:
        print(json.dumps(catalog(), indent=2, sort_keys=True))
        return 0
    if args.llm_catalog:
        print(json.dumps(renderer.llm_catalog(), indent=2, sort_keys=True))
        return 0
    if args.list_effects:
        for effect in renderer.effects():
            print(effect)
        return 0
    if not args.uri:
        parser.error("URI is required unless --list-effects, --catalog, or --llm-catalog is used")
    sound = renderer.render_uri(args.uri)
    path = sound.write_wav(args.output)
    print(f"{path} ({sound.duration:.3f}s, {sound.sample_rate} Hz)")
    return 0
