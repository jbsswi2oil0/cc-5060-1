"""Audio file I/O (WAV/FLAC/OGG via soundfile/libsndfile)."""

from __future__ import annotations

import numpy as np
import soundfile as sf


def load_audio(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=True, dtype="float64")
    return data, int(sr)


def save_audio(path: str, samples: np.ndarray, sr: int, subtype: str | None = None) -> None:
    if samples.ndim == 1:
        samples = samples[:, None]
    # No implicit normalization: float WAV keeps >1.0 values so applied gains
    # remain exactly verifiable after export/reload.
    sf.write(path, samples, sr, subtype=subtype)


def write_float_wav(path: str, samples: np.ndarray, sr: int) -> None:
    save_audio(path, samples, sr, subtype="FLOAT")
