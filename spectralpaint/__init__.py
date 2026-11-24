"""Spectral smearing audio editor package."""

from .dsp import STFTConfig, istft, stft, roundtrip_error
from .engine import EditSession
from .mask import Stroke

__all__ = ["STFTConfig", "stft", "istft", "roundtrip_error",
           "EditSession", "Stroke"]
