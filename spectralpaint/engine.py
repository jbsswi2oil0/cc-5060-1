"""Spectral editing engine.

The engine never mutates the original signal.  It keeps:
  * the pristine time-domain signal (and its per-channel complex STFT),
  * an ordered list of :class:`~spectralpaint.mask.Stroke` edits.

Reconstruction re-applies *all* strokes from scratch to the pristine complex
spectrum, so an undo removes a stroke and rebuilds exactly -- it never multiplies
by an inverse gain that could fail to recover the original numbers.

Two cross-channel strategies are offered for stereo material:

``phase_locked``
    Both channels are scaled by the *same real-valued* gain field.  Magnitudes
    change while the inter-channel phase difference arg(L)-arg(R) is preserved
    cell by cell.  This is the right choice for narrow-band edits that should
    not move the stereo image.

``coherent_joint``
    Time-frequency cells are split into coherent (common/center content) and
    incoherent (ambient/channel-specific) content using a time-frequency
    coherence estimate.  Coherent cells get a single jointly re-estimated gain
    (average energy across channels), keeping the pan/phase relation; incoherent
    cells are re-estimated independently per channel.

Real-signal conjugacy is guaranteed structurally: edits are applied only to the
rfft (non-negative frequency) representation and irfft regenerates the missing
half with the required X[k] = conj(X[N-k]) symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import STFTConfig, freq_bin_centers, frame_times, istft, stft
from .mask import Stroke, composite_linear_gain


@dataclass
class EditSession:
    samples: np.ndarray          # (n_samples, n_channels) float64
    sr: int
    cfg: STFTConfig
    strokes: list[Stroke] = field(default_factory=list)
    strategy: str = "phase_locked"
    coherence_threshold: float = 0.6
    coherence_smooth: tuple[int, int] = (3, 3)  # (time, freq) smoothing kernel

    def __post_init__(self):
        if self.samples.ndim == 1:
            self.samples = self.samples[:, None]
        self.samples = self.samples.astype(np.float64, copy=False)
        self._base_specs = [stft(self.samples[:, c], self.cfg)
                            for c in range(self.n_channels)]
        self._times = frame_times(self.cfg, self._base_specs[0].shape[0], self.sr)
        self._freqs = freq_bin_centers(self.cfg, self.sr)

    # ------------------------------------------------------------------ basic
    @property
    def n_channels(self) -> int:
        return self.samples.shape[1]

    @property
    def n_frames(self) -> int:
        return self._base_specs[0].shape[0]

    @property
    def n_bins(self) -> int:
        return self._base_specs[0].shape[1]

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def freqs(self) -> np.ndarray:
        return self._freqs

    # ---------------------------------------------------------------- editing
    def add_stroke(self, stroke: Stroke) -> int:
        self.strokes.append(stroke)
        return len(self.strokes) - 1

    def undo(self) -> Stroke | None:
        """Exact undo: drop the last stroke; reconstruction returns to prior state."""
        if not self.strokes:
            return None
        return self.strokes.pop()

    def clear_edits(self) -> None:
        self.strokes.clear()

    def gain_field(self) -> np.ndarray:
        """Linear gain rasterized on the physical STFT grid (n_frames, n_bins)."""
        return composite_linear_gain(self.strokes, self._times, self._freqs)

    def mask_region(self, n_sigma: float = 3.0) -> np.ndarray:
        """Boolean grid covered by selections expanded by n_sigma of feather."""
        from .mask import effective_bounds
        b = effective_bounds(self.strokes, n_sigma=n_sigma)
        if b is None:
            return np.zeros((self.n_frames, self.n_bins), dtype=bool)
        t_in = (self._times >= b["t0"]) & (self._times <= b["t1"])
        f_in = (self._freqs >= b["f0"]) & (self._freqs <= b["f1"])
        return t_in[:, None] & f_in[None, :]

    # -------------------------------------------------------------- coherence
    def coherence(self) -> np.ndarray:
        """Time-frequency magnitude-squared coherence between L and R.

        gamma^2(t,f) = |S_lr|^2 / (S_ll * S_rr) with a small 2-D uniform
        smoother over time and frequency so the estimate is not 1 everywhere
        (which raw single-frame cross-spectra would be).
        """
        if self.n_channels < 2:
            return np.ones((self.n_frames, self.n_bins))
        L, R = self._base_specs[0], self._base_specs[1]
        s_lr = _box_smooth(L * np.conj(R), *self.coherence_smooth)
        s_ll = _box_smooth(np.abs(L) ** 2, *self.coherence_smooth)
        s_rr = _box_smooth(np.abs(R) ** 2, *self.coherence_smooth)
        denom = s_ll * s_rr
        gamma2 = np.zeros_like(denom)
        nz = denom > 1e-20
        gamma2[nz] = np.abs(s_lr[nz]) ** 2 / denom[nz]
        return np.clip(gamma2, 0.0, 1.0)

    # ------------------------------------------------------------- rebuilding
    def edited_spectra(self) -> list[np.ndarray]:
        """Apply strokes to pristine complex spectra under the chosen strategy."""
        g = self.gain_field()  # (T, F) real, >= 0
        if self.n_channels == 1:
            return [self._base_specs[0] * g]

        if self.strategy == "phase_locked":
            # Identical real multiplier on both channels: preserves arg(L)-arg(R).
            return [spec * g for spec in self._base_specs]

        if self.strategy == "coherent_joint":
            return self._coherent_joint(g)

        raise ValueError(f"unknown strategy {self.strategy!r}")

    def _coherent_joint(self, g: np.ndarray) -> list[np.ndarray]:
        L, R = self._base_specs[0], self._base_specs[1]
        gamma2 = self.coherence()
        coherent = gamma2 >= self.coherence_threshold  # bool (T, F)

        # Joint re-estimation on coherent cells: a common gain derived from the
        # mean energy of the two channels, applied identically (same real factor
        # per channel) so the pan/phase relation is untouched.
        p_l = np.abs(L) ** 2
        p_r = np.abs(R) ** 2
        mean_power = 0.5 * (p_l + p_r)
        num = _box_smooth((g ** 2) * mean_power, *self.coherence_smooth)
        den = _box_smooth(mean_power, *self.coherence_smooth)
        ratio = np.ones_like(mean_power)
        ok = den > 1e-20
        ratio[ok] = num[ok] / den[ok]
        joint_g = np.sqrt(np.clip(ratio, 0.0, None))

        gL = np.where(coherent, joint_g, g)
        gR = np.where(coherent, joint_g, g)
        # Incoherent cells: independent per-channel re-estimation using each
        # channel's own energy (the requested gain directly, since it already
        # encodes the painter's intent per cell).
        return [L * gL, R * gR]

    def rebuild(self) -> np.ndarray:
        """Inverse-transform edited spectra and overlap-add back to waveform."""
        specs = self.edited_spectra()
        n = self.samples.shape[0]
        out = np.empty((n, self.n_channels), dtype=np.float64)
        for c, spec in enumerate(specs):
            out[:, c] = istft(spec, self.cfg, n)
        return out

    def pristine(self) -> np.ndarray:
        return self.samples.copy()


def _box_smooth(x: np.ndarray, nt: int, nf: int) -> np.ndarray:
    """Uniform 2-D moving average with edge padding ('same' shape)."""
    if nt <= 1 and nf <= 1:
        return x.copy()
    pad_t, pad_f = nt // 2, nf // 2
    xp = np.pad(x, ((pad_t, pad_t), (pad_f, pad_f)), mode="edge")
    win = np.ones((nt, nf), dtype=np.float64) / (nt * nf)
    from scipy.signal import fftconvolve
    return fftconvolve(xp, win, mode="same")[pad_t:pad_t + x.shape[0],
                                              pad_f:pad_f + x.shape[1]]
