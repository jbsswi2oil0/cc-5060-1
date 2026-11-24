"""Spectral smearing editor: core STFT / ISTFT and OLA normalization.

Design invariants
-----------------
* Analysis and synthesis use the *same* window; reconstruction normalizes by
  the squared-window overlap-add envelope, which is the correct adjoint
  weighting.  When (window, hop) satisfies COLA the envelope is constant and
  reconstruction is exact (round-trip error at floating-point level).
* All transforms operate on the *complex* spectrum.  Real signals are handled
  with rfft/irfft so the conjugate symmetry X[k] = conj(X[N-k]) is maintained
  automatically -- editing only the displayed (non-negative) half can never
  break the conjugacy of the reconstructed real waveform.
* Frames are computed on centered FFT windows with explicit zero padding, so
  frame count and boundary behavior are fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_window(name: str, n_fft: int) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n_fft + 1)[:-1].astype(np.float64)
    if name == "hamming":
        return np.hamming(n_fft).astype(np.float64)
    if name == "rect":
        return np.ones(n_fft, dtype=np.float64)
    if name == "sine":
        # root-Hann: squared window is Hann -> exact COLA at hop = n_fft/2
        return np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float64)
    raise ValueError(f"unknown window {name!r}")


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 2048
    hop: int = 512
    window: str = "hann"

    def win(self) -> np.ndarray:
        return make_window(self.window, self.n_fft)

    @property
    def sample_rate_hint(self) -> float:
        return 1.0


def frame_count(n_samples: int, n_fft: int, hop: int) -> int:
    """Number of centered analysis frames covering n_samples.

    Frame t starts at t*hop - n_fft//2; we keep frames whose window overlaps
    the signal, matching the padding produced by :func:`pad_signal`.
    """
    if n_samples <= 0:
        return 0
    return int(np.ceil(n_samples / hop)) + 1


def pad_signal(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Pad a 1-D signal so centered frames cover it fully on both ends."""
    n = x.shape[-1]
    n_frames = frame_count(n, n_fft, hop)
    # Last frame start (in padded coords): n_fft//2 + (n_frames-1)*hop
    last_end = n_fft // 2 + (n_frames - 1) * hop + n_fft
    needed = last_end - n_fft // 2  # required padded length
    pad_right = max(0, needed - n)
    return np.pad(x, (n_fft // 2, pad_right), mode="constant")


def stft(x: np.ndarray, cfg: STFTConfig) -> np.ndarray:
    """Complex STFT of a real 1-D signal.

    Returns shape (n_frames, n_bins) with n_bins = n_fft//2 + 1.
    """
    if x.ndim != 1:
        raise ValueError("stft expects a 1-D real signal")
    n_fft, hop = cfg.n_fft, cfg.hop
    if hop <= 0 or hop > n_fft:
        raise ValueError("hop must be in (0, n_fft]")
    win = cfg.win()
    xp = pad_signal(x.astype(np.float64, copy=False), n_fft, hop)
    n_frames = frame_count(x.shape[-1], n_fft, hop)
    n_bins = n_fft // 2 + 1
    spec = np.empty((n_frames, n_bins), dtype=np.complex128)
    for t in range(n_frames):
        start = t * hop
        frame = xp[start:start + n_fft]
        if frame.shape[0] < n_fft:  # pragma: no cover - padding prevents this
            frame = np.pad(frame, (0, n_fft - frame.shape[0]))
        spec[t] = np.fft.rfft(frame * win, n=n_fft)
    return spec


def istft(spec: np.ndarray, cfg: STFTConfig, n_samples: int) -> np.ndarray:
    """Inverse STFT with squared-window overlap-add normalization.

    Parameters
    ----------
    spec : (n_frames, n_bins) complex array (rfft layout).
    n_samples : original signal length, used to trim the output.
    """
    n_fft, hop = cfg.n_fft, cfg.hop
    win = cfg.win()
    n_frames = spec.shape[0]
    out_len = n_fft // 2 + (n_frames - 1) * hop + n_fft
    y = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    w2 = win * win
    for t in range(n_frames):
        frame = np.fft.irfft(spec[t], n=n_fft)
        start = t * hop
        end = start + n_fft
        y[start:end] += frame * win
        wsum[start:end] += w2
    # Adjoint normalization.  Under COLA wsum is (near) constant; the per-sample
    # divide also keeps boundaries correct where fewer frames overlap.
    nz = wsum > 1e-12
    y[nz] /= wsum[nz]
    start_trim = n_fft // 2
    return y[start_trim:start_trim + n_samples]


def roundtrip_error(x: np.ndarray, cfg: STFTConfig) -> dict:
    """Diagnostic: reconstruct without any edit and measure the error."""
    spec = stft(x, cfg)
    xr = istft(spec, cfg, x.shape[-1])
    err = xr - x
    scale = np.sqrt(np.mean(x ** 2)) + 1e-30
    return {
        "rms": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
        "relative_rms_db": float(20 * np.log10(np.sqrt(np.mean(err ** 2)) / scale)),
        "cola_envelope_std": float(
            np.std(_ola_envelope(cfg, frame_count(x.shape[-1], cfg.n_fft, cfg.hop)))
        ),
    }


def _ola_envelope(cfg: STFTConfig, n_frames: int) -> np.ndarray:
    """Squared-window overlap envelope (COLA diagnostic)."""
    n_fft, hop = cfg.n_fft, cfg.hop
    w2 = cfg.win() ** 2
    length = n_fft // 2 + (max(n_frames, 1) - 1) * hop + n_fft
    env = np.zeros(length)
    for t in range(n_frames):
        env[t * hop:t * hop + n_fft] += w2
    return env[n_fft // 2:]


def freq_bin_centers(cfg: STFTConfig, sr: float) -> np.ndarray:
    """Physical frequency (Hz) of each rfft bin -- independent of display zoom."""
    return np.fft.rfftfreq(cfg.n_fft, d=1.0 / sr)


def frame_times(cfg: STFTConfig, n_frames: int, sr: float) -> np.ndarray:
    """Physical center time (s) of each analysis frame."""
    return (np.arange(n_frames) * cfg.hop) / sr
