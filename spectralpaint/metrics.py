"""Quantitative analysis of edited vs. pristine audio.

Metrics
-------
pan_shift_db
    Per-cell stereo pan change, computed from the mid/side ratio in dB.  For
    cell k: pan = 10*log10(|R|^2 / |L|^2); the metric is the change versus the
    pristine signal.  A correct phase-locked narrow-band edit gives ~0 dB inside
    the edited region.

leakage_untouched
    Energy that appears in time-frequency cells the mask should not have
    touched (outside edge + 3-sigma feather), expressed relative to the
    pristine energy in those cells.  Non-zero values quantify spectral leakage
    / cross-cell contamination caused by the windowed transform and edges.

boundary_residue
    Energy change concentrated in the feather ring around each selection,
    split per axis, useful to verify the smoothing width behaves as specified.

roundtrip
    Reconstruction error when no edit is applied (must be ~machine precision
    for a COLA window/hop pair).
"""

from __future__ import annotations

import numpy as np

from .dsp import stft
from .mask import effective_bounds


def pan_field(spec_l: np.ndarray, spec_r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pan position in dB: positive = right, negative = left."""
    return 10.0 * np.log10((np.abs(spec_r) ** 2 + eps) / (np.abs(spec_l) ** 2 + eps))


def pan_shift(spec_l0: np.ndarray, spec_r0: np.ndarray,
              spec_l1: np.ndarray, spec_r1: np.ndarray,
              region: np.ndarray | None = None) -> dict:
    p0 = pan_field(spec_l0, spec_r0)
    p1 = pan_field(spec_l1, spec_r1)
    delta = p1 - p0
    # Pan is only meaningful where there is actual signal energy.
    energy = np.abs(spec_l0) ** 2 + np.abs(spec_r0) ** 2
    active = energy > (1e-6 * energy.max(initial=1.0))
    if region is not None:
        active &= region
    if not active.any():
        return {"pan_shift_rms_db": 0.0, "pan_shift_max_abs_db": 0.0, "cells": 0}
    d = delta[active]
    return {
        "pan_shift_rms_db": float(np.sqrt(np.mean(d ** 2))),
        "pan_shift_max_abs_db": float(np.max(np.abs(d))),
        "mean_pan_before_db": float(np.mean(p0[active])),
        "mean_pan_after_db": float(np.mean(p1[active])),
        "cells": int(active.sum()),
    }


def interchannel_phase_diff(spec_l: np.ndarray, spec_r: np.ndarray) -> np.ndarray:
    return np.angle(spec_l) - np.angle(spec_r)


def phase_diff_change(spec_l0, spec_r0, spec_l1, spec_r1,
                      region: np.ndarray | None = None) -> dict:
    d0 = interchannel_phase_diff(spec_l0, spec_r0)
    d1 = interchannel_phase_diff(spec_l1, spec_r1)
    # wrap to (-pi, pi]
    change = np.angle(np.exp(1j * (d1 - d0)))
    energy = np.abs(spec_l0) ** 2 + np.abs(spec_r0) ** 2
    active = energy > (1e-6 * energy.max(initial=1.0))
    if region is not None:
        active &= region
    if not active.any():
        return {"phase_diff_change_rms_rad": 0.0, "cells": 0}
    c = change[active]
    return {"phase_diff_change_rms_rad": float(np.sqrt(np.mean(c ** 2))),
            "phase_diff_change_max_abs_rad": float(np.max(np.abs(c))),
            "cells": int(active.sum())}


def untouched_region(times: np.ndarray, freqs: np.ndarray,
                     strokes, n_sigma: float = 6.0) -> np.ndarray:
    """Boolean mask (T,F) of cells well outside every edited selection.

    The default 6-sigma expansion puts the Gaussian feather tail below 1e-8 at
    the boundary, so any energy change measured here reflects genuine spectral
    leakage / OLA cross-cell effects rather than the intended smoothing.
    """
    b = effective_bounds(strokes, n_sigma=n_sigma)
    if b is None:
        return np.ones((times.size, freqs.size), dtype=bool)
    t_in = (times >= b["t0"]) & (times <= b["t1"])
    f_in = (freqs >= b["f0"]) & (freqs <= b["f1"])
    inside = t_in[:, None] & f_in[None, :]
    return ~inside


def leakage_untouched(specs0: list[np.ndarray], specs1: list[np.ndarray],
                      untouched: np.ndarray) -> dict:
    """Relative energy change in cells that should be untouched."""
    rows = []
    for c, (a, b) in enumerate(zip(specs0, specs1)):
        e0 = np.abs(a[untouched]) ** 2
        e1 = np.abs(b[untouched]) ** 2
        total0 = np.abs(a) ** 2
        # absolute leaked energy relative to total pristine energy
        leak = np.sum(np.abs(b - a) ** 2, where=untouched) / (np.sum(total0) + 1e-30)
        rows.append({"channel": c,
                     "leak_energy_ratio": float(leak),
                     "untouched_energy_change_db": float(10 * np.log10((e1.sum() + 1e-30) / (e0.sum() + 1e-30)))})
    return {"per_channel": rows,
            "max_leak_energy_ratio": float(max(r["leak_energy_ratio"] for r in rows))}


def boundary_ring(times: np.ndarray, freqs: np.ndarray, strokes,
                  inner_sigma: float = 1.0, outer_sigma: float = 3.0) -> np.ndarray:
    """Cells in the feather band: inside outer (3-sigma) but outside inner (1-sigma)."""
    inner = effective_bounds(strokes, n_sigma=inner_sigma)
    outer = effective_bounds(strokes, n_sigma=outer_sigma)
    if outer is None:
        return np.zeros((times.size, freqs.size), dtype=bool)

    def box(b):
        t_in = (times >= b["t0"]) & (times <= b["t1"])
        f_in = (freqs >= b["f0"]) & (freqs <= b["f1"])
        return t_in[:, None] & f_in[None, :]

    return box(outer) & ~box(inner)


def boundary_residue(specs0, specs1, ring: np.ndarray) -> dict:
    out = []
    for c, (a, b) in enumerate(zip(specs0, specs1)):
        if not ring.any():
            out.append({"channel": c, "ring_energy_change_db": 0.0})
            continue
        e0 = np.sum(np.abs(a[ring]) ** 2)
        e1 = np.sum(np.abs(b[ring]) ** 2)
        out.append({"channel": c,
                    "ring_energy_change_db": float(10 * np.log10((e1 + 1e-30) / (e0 + 1e-30))),
                    "ring_pristine_energy": float(e0),
                    "ring_edited_energy": float(e1)})
    return {"per_channel": out}


def time_domain_delta(x0: np.ndarray, x1: np.ndarray) -> dict:
    err = x1 - x0
    scale = np.sqrt(np.mean(x0 ** 2)) + 1e-30
    return {
        "rms": float(np.sqrt(np.mean(err ** 2))),
        "peak_abs": float(np.max(np.abs(err))),
        "relative_rms_db": float(20 * np.log10(np.sqrt(np.mean(err ** 2)) / scale)),
    }


def full_report(session, edited_samples: np.ndarray) -> dict:
    """Run every metric for an edited session (used by UI and tests).

    Pan/phase are reported twice:
      * ``pan``/``phase``: measured on the *exact edited complex spectra* before
        the inverse transform -- these isolate engine behavior and are ~0 for a
        correct phase-locked real gain;
      * ``pan_reanalyzed``/``phase_reanalyzed``: measured after OLA
        reconstruction and a fresh STFT, including window re-analysis effects
        at selection boundaries.
    """
    cfg = session.cfg
    specs0 = session._base_specs
    exact_edited = session.edited_spectra()
    specs1 = [stft(edited_samples[:, c], cfg) for c in range(session.n_channels)]
    untouched = untouched_region(session.times, session.freqs, session.strokes)
    report = {
        "n_strokes": len(session.strokes),
        "strategy": session.strategy,
        "leakage": leakage_untouched(specs0, specs1, untouched),
        "boundary": boundary_residue(specs0, specs1,
                                    boundary_ring(session.times, session.freqs, session.strokes)),
        "time_domain": time_domain_delta(session.samples, edited_samples),
        "effective_bounds": effective_bounds(session.strokes),
    }
    if session.n_channels >= 2:
        # region inside the hard selection (0-sigma bounds) for pan/phase checks
        hard = ~untouched_region(session.times, session.freqs, session.strokes, n_sigma=0.0)
        report["pan"] = pan_shift(specs0[0], specs0[1], exact_edited[0], exact_edited[1], hard)
        report["phase"] = phase_diff_change(specs0[0], specs0[1], exact_edited[0], exact_edited[1], hard)
        report["pan_reanalyzed"] = pan_shift(specs0[0], specs0[1], specs1[0], specs1[1], hard)
        report["phase_reanalyzed"] = phase_diff_change(specs0[0], specs0[1], specs1[0], specs1[1], hard)
    return report
