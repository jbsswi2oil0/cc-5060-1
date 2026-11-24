"""Deterministic test-signal generators: dual-tone, sweep, transient mixture.

All generators return float64 arrays shaped (n_samples, n_channels) and are
fully deterministic, so the test suite can assert exact spectral behavior
(e.g. a narrow-band edit on one tone must not move the other tone).
"""

from __future__ import annotations

import numpy as np


def _pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Pan a mono signal: pan in [-1, 1], -1 left, 0 center, +1 right."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * np.pi / 4.0
    left = mono * np.cos(angle)
    right = mono * np.sin(angle)
    return np.stack([left, right], axis=1)


def dual_tone(sr: int = 44100, duration: float = 2.0,
              f1: float = 440.0, f2: float = 1760.0,
              amp1: float = 0.5, amp2: float = 0.35,
              pan1: float = -0.6, pan2: float = 0.6) -> tuple[np.ndarray, int]:
    """Two steady tones at distinct frequencies and pan positions."""
    t = np.arange(int(sr * duration)) / sr
    m1 = amp1 * np.sin(2 * np.pi * f1 * t)
    m2 = amp2 * np.sin(2 * np.pi * f2 * t)
    stereo = _pan_stereo(m1, pan1) + _pan_stereo(m2, pan2)
    return stereo / np.max(np.abs(stereo)) * 0.9, sr


def sweep(sr: int = 44100, duration: float = 3.0,
          f0: float = 200.0, f1: float = 4000.0,
          pan: float = 0.0) -> tuple[np.ndarray, int]:
    """Linear chirp with constant amplitude, optionally panned."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    k = (f1 - f0) / duration
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    mono = 0.7 * np.sin(phase)
    env = np.minimum(1.0, np.minimum(t / 0.02, (duration - t) / 0.02))
    mono *= np.clip(env, 0.0, 1.0)
    return _pan_stereo(mono, pan), sr


def transient_mix(sr: int = 44100, duration: float = 2.0,
                  tone_f: float = 1000.0, n_transients: int = 4,
                  pan: float = -0.3) -> tuple[np.ndarray, int]:
    """Steady tone plus short broadband clicks at known times (boundary test)."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    mono = 0.25 * np.sin(2 * np.pi * tone_f * t)
    rng = np.random.default_rng(7)
    click_times = np.linspace(0.3, duration - 0.3, n_transients)
    for ct in click_times:
        idx = int(ct * sr)
        length = int(0.008 * sr)  # 8 ms click
        env = np.exp(-np.linspace(0, 6, length))
        click = 0.8 * env * rng.choice([-1, 1]) * np.ones(length)
        end = min(n, idx + length)
        mono[idx:end] += click[:end - idx]
    return _pan_stereo(mono, pan), sr


def mixed_sample(sr: int = 44100, duration: float = 4.0) -> tuple[np.ndarray, int]:
    """Combined signal: low tone (left), high tone (right), chirp, transients."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    out = np.zeros((n, 2))
    # low tone, panned left
    out += _pan_stereo(0.3 * np.sin(2 * np.pi * 300 * t), -0.7)
    # high tone, panned right
    out += _pan_stereo(0.25 * np.sin(2 * np.pi * 2500 * t), 0.7)
    # upward chirp in center
    k = (3000 - 300) / duration
    phase = 2 * np.pi * (300 * t + 0.5 * k * t ** 2)
    out += _pan_stereo(0.18 * np.sin(phase), 0.0)
    # three clicks alternating pan
    rng = np.random.default_rng(11)
    for i, ct in enumerate(np.linspace(0.5, duration - 0.5, 3)):
        idx = int(ct * sr)
        length = int(0.006 * sr)
        env = np.exp(-np.linspace(0, 6, length))
        click = 0.7 * env * rng.choice([-1, 1])
        mono = np.zeros(n)
        mono[idx:idx + length] += click
        out += _pan_stereo(mono, -0.8 if i % 2 == 0 else 0.8)
    peak = np.max(np.abs(out))
    return out / peak * 0.9, sr
