"""Selection *storage* representations for spectral edits.

Two candidate representations are implemented side by side so the experiment
in ``scripts/run_selection_experiment.py`` can compare them on the *same*
inverse-transform prototype (:func:`spectralpaint.dsp.istft`):

``CurveSelection``
    Continuous time-frequency boundary stored as curves in physical units
    (seconds / Hz): two polylines ``f_lo(t)``, ``f_hi(t)`` plus gain and
    feather widths.  Rasterization onto an STFT grid happens at *evaluation*
    time, on whatever grid is current.  A rectangle (the editor's Stroke) is
    the degenerate case with two control points per curve.

``RasterizedMask``
    Coefficient mask: the gain field frozen on the STFT grid of the window
    parameters (n_fft / hop / window) it was created with, stored together
    with those parameters.  Re-opening it on a different grid requires an
    explicit policy:

    ``strict``
        Refuse: the mask is only meaningful on its original grid.
    ``pixel``
        Index-preserving ("pixel-fixed"): cell (i, j) stays cell (i, j).
        The physical meaning of each cell shifts when the grid changes --
        this is exactly the failure mode the experiment quantifies.
    ``resample``
        Interpolate the stored field in physical (t, f) coordinates onto the
        new grid.  The band lands approximately right, but the boundary was
        quantized to the old grid at creation time and cannot be recovered.

Neither representation is the display grid: canvas pixels never enter either
one.  Coordinates are seconds / Hz (curve) or STFT cells bound to explicit
window parameters (mask).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import STFTConfig, frame_times, freq_bin_centers
from .mask import Stroke, _half_support, composite_gain_db


# ------------------------------------------------------------------ curves --
@dataclass
class CurveSelection:
    """Continuous selection boundary: f_lo(t), f_hi(t) polylines in s / Hz."""

    t_ctrl: tuple[float, ...]
    f_lo: tuple[float, ...]
    f_hi: tuple[float, ...]
    gain_db: float
    feather_t: float = 0.03
    feather_f: float = 60.0

    def __post_init__(self):
        if not (len(self.t_ctrl) == len(self.f_lo) == len(self.f_hi)):
            raise ValueError("control point arrays must have equal length")
        if len(self.t_ctrl) < 2:
            raise ValueError("need at least two control points")
        if any(b < a for a, b in zip(self.f_lo, self.f_hi)):
            raise ValueError("f_lo must be <= f_hi at every control point")

    @classmethod
    def from_stroke(cls, s: Stroke) -> "CurveSelection":
        """A rectangular stroke is the degenerate curve: constant edges."""
        n = s.normalized()
        return cls((n.t0, n.t1), (n.f0, n.f0), (n.f1, n.f1),
                   n.gain_db, n.feather_t, n.feather_f)

    def gain_db_field(self, times: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Evaluate the continuous boundary on an arbitrary STFT grid (dB)."""
        t_ctrl = np.asarray(self.t_ctrl, dtype=np.float64)
        lo_t = np.interp(times, t_ctrl, np.asarray(self.f_lo, dtype=np.float64))
        hi_t = np.interp(times, t_ctrl, np.asarray(self.f_hi, dtype=np.float64))
        wt = _half_support(times, float(t_ctrl[0]), float(t_ctrl[-1]),
                           self.feather_t)                      # (n_frames,)
        f = freqs[None, :]
        if self.feather_f <= 0.0:
            wf = ((f >= lo_t[:, None]) & (f <= hi_t[:, None])).astype(np.float64)
        else:
            d = np.maximum(0.0, np.maximum(lo_t[:, None] - f, f - hi_t[:, None]))
            wf = np.exp(-0.5 * (d / self.feather_f) ** 2)
        return self.gain_db * (wt[:, None] * wf)

    @property
    def storage_bytes(self) -> int:
        """Persistent size: 3 floats per control point + 3 scalars."""
        return 8 * (3 * len(self.t_ctrl) + 3)


# -------------------------------------------------------------- coefficient --
@dataclass
class RasterizedMask:
    """Gain mask frozen on the STFT grid of ``cfg`` (stored with it)."""

    gain_db: np.ndarray           # (n_frames, n_bins) at creation window
    cfg: STFTConfig
    sr: int
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_strokes(cls, strokes: list[Stroke], cfg: STFTConfig, sr: int,
                     n_frames: int) -> "RasterizedMask":
        times = frame_times(cfg, n_frames, sr)
        freqs = freq_bin_centers(cfg, sr)
        field = composite_gain_db(strokes, times, freqs)
        return cls(field, cfg, sr, meta={"n_strokes": len(strokes)})

    def gain_db_field(self, cfg: STFTConfig, n_frames: int,
                      policy: str = "strict") -> np.ndarray:
        """Re-open the stored mask on a (possibly different) grid."""
        n_bins = cfg.n_fft // 2 + 1
        same_window = (cfg.n_fft == self.cfg.n_fft
                       and cfg.hop == self.cfg.hop
                       and cfg.window == self.cfg.window)
        if policy == "strict":
            if not same_window or n_frames != self.gain_db.shape[0]:
                raise ValueError(
                    "coefficient mask is bound to its original window "
                    f"parameters (n_fft={self.cfg.n_fft}, hop={self.cfg.hop}, "
                    f"window={self.cfg.window!r}, frames={self.gain_db.shape[0]}); "
                    f"requested n_fft={cfg.n_fft}, hop={cfg.hop}, "
                    f"window={cfg.window!r}, frames={n_frames}")
            return self.gain_db.copy()

        if policy == "pixel":
            # Index-preserving: cell (i, j) keeps its *index*, not its physical
            # (t, f) meaning.  Where the new grid is larger, gain is 0 dB.
            out = np.zeros((n_frames, n_bins), dtype=np.float64)
            nt = min(n_frames, self.gain_db.shape[0])
            nf = min(n_bins, self.gain_db.shape[1])
            out[:nt, :nf] = self.gain_db[:nt, :nf]
            return out

        if policy == "resample":
            # Interpolate the frozen field in physical coordinates.  The band
            # lands approximately right, but the boundary was quantized to the
            # old grid at creation time and stays quantized.
            t_old = frame_times(self.cfg, self.gain_db.shape[0], self.sr)
            f_old = freq_bin_centers(self.cfg, self.sr)
            t_new = frame_times(cfg, n_frames, self.sr)
            f_new = freq_bin_centers(cfg, self.sr)
            # separable linear interpolation; outside the stored grid -> 0 dB
            tmp = np.empty((self.gain_db.shape[0], f_new.size))
            for i in range(self.gain_db.shape[0]):
                tmp[i] = np.interp(f_new, f_old, self.gain_db[i],
                                   left=0.0, right=0.0)
            out = np.empty((t_new.size, f_new.size))
            for j in range(f_new.size):
                out[:, j] = np.interp(t_new, t_old, tmp[:, j],
                                      left=0.0, right=0.0)
            return out

        raise ValueError(f"unknown policy {policy!r}")

    @property
    def storage_bytes(self) -> int:
        return self.gain_db.nbytes


# --------------------------------------------------------------- measurement --
def half_gain_bounds(field_db: np.ndarray, times: np.ndarray,
                     freqs: np.ndarray, gain_db: float) -> dict | None:
    """Physical (s / Hz) bounds where the field crosses half of ``gain_db``.

    This is the *measured* mask boundary on whatever grid ``field_db`` lives
    on -- used to quantify how far a stored selection's boundary lands from
    its intent after storage and (possibly) a window change.
    """
    if gain_db == 0.0:
        raise ValueError("gain_db must be non-zero to measure a boundary")
    half = gain_db / 2.0
    if gain_db < 0:
        proj_f = field_db.min(axis=0)
        proj_t = field_db.min(axis=1)
        inside_f = proj_f <= half
        inside_t = proj_t <= half
    else:
        proj_f = field_db.max(axis=0)
        proj_t = field_db.max(axis=1)
        inside_f = proj_f >= half
        inside_t = proj_t >= half
    if not inside_f.any() or not inside_t.any():
        return None
    fi = np.flatnonzero(inside_f)
    ti = np.flatnonzero(inside_t)
    return {"f_lo": float(freqs[fi[0]]), "f_hi": float(freqs[fi[-1]]),
            "t_lo": float(times[ti[0]]), "t_hi": float(times[ti[-1]])}
