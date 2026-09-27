"""Compatibility re-exports for deterministic random substreams."""

from .rng import RandomStream, event_rng

__all__ = ["RandomStream", "event_rng"]
