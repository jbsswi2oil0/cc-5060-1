"""Gain masks defined in *physical* units (seconds, Hz) with soft edges.

The editor strokes are stored as rectangles in (time, frequency) space, never
as canvas pixels.  Rasterization to the STFT grid happens through bin center
frequencies and frame center times, so zooming / resizing the display does not
change which time-frequency cells are affected or by how much.

Each stroke carries:
    t0, t1, f0, f1 : selection bounds in seconds / Hz
    gain_db       : applied gain inside the selection
    feather_t     : Gaussian smoothing width in seconds (1-sigma at edge)
    feather_f     : Gaussian smoothing width in Hz

The resulting per-cell gain is a smooth field in dB: the hard selection gets a
Gaussian roll-off, then gain is converted to a linear multiplier.  Feather
width is therefore interpretable ("sigma = 40 ms / 80 Hz") instead of being
"some number of pixels".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stroke:
    t0: float
    t1: float
    f0: float
    f1: float
    gain_db: float
    feather_t: float = 0.03   # seconds, 1-sigma
    feather_f: float = 60.0   # Hz, 1-sigma
    kind: str = "rect"

    def normalized(self) -> "Stroke":
        return Stroke(
            min(self.t0, self.t1), max(self.t0, self.t1),
            min(self.f0, self.f1), max(self.f0, self.f1),
            self.gain_db, self.feather_t, self.feather_f, self.kind,
        )


def _half_support(center: np.ndarray, lo: float, hi: float, sigma: float) -> np.ndarray:
    """Smooth membership in [lo, hi]: 1 inside, Gaussian roll-off outside.

    Distance is measured from the *nearest* edge; sigma is the 1-standard-
    deviation width.  sigma == 0 gives a hard mask.
    """
    if sigma <= 0.0:
        return ((center >= lo) & (center <= hi)).astype(np.float64)
    d = np.maximum(0.0, np.maximum(lo - center, center - hi))
    return np.exp(-0.5 * (d / sigma) ** 2)


def stroke_gain_db(stroke: Stroke, times: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Rasterize one stroke onto the (n_frames, n_bins) grid, in dB."""
    s = stroke.normalized()
    wt = _half_support(times, s.t0, s.t1, s.feather_t)          # (n_frames,)
    wf = _half_support(freqs, s.f0, s.f1, s.feather_f)          # (n_bins,)
    # Product of the two soft memberships => elliptical-style soft rectangle
    return s.gain_db * (wt[:, None] * wf[None, :])


def composite_gain_db(strokes: list[Stroke], times: np.ndarray,
                      freqs: np.ndarray) -> np.ndarray:
    """Combine strokes by *summing* their dB fields.

    Gains compose multiplicatively in linear space, i.e. additively in dB,
    which lets overlapping strokes behave predictably.
    """
    g = np.zeros((times.shape[0], freqs.shape[0]), dtype=np.float64)
    for s in strokes:
        g += stroke_gain_db(s, times, freqs)
    return g


def composite_linear_gain(strokes: list[Stroke], times: np.ndarray,
                          freqs: np.ndarray) -> np.ndarray:
    return 10.0 ** (composite_gain_db(strokes, times, freqs) / 20.0)


def effective_bounds(strokes: list[Stroke], n_sigma: float = 3.0) -> dict | None:
    """Report the practical affected region (edge + n_sigma * feather).

    Used by the UI to display the actual scope of edits and by tests to check
    that untouched cells stay untouched.
    """
    if not strokes:
        return None
    t0 = min(s.normalized().t0 - n_sigma * s.feather_t for s in strokes)
    t1 = max(s.normalized().t1 + n_sigma * s.feather_t for s in strokes)
    f0 = min(s.normalized().f0 - n_sigma * s.feather_f for s in strokes)
    f1 = max(s.normalized().f1 + n_sigma * s.feather_f for s in strokes)
    return {"t0": max(0.0, t0), "t1": t1, "f0": max(0.0, f0), "f1": f1}
