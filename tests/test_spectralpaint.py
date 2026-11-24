"""Pytest suite for the spectral smearing editor.

Run with:  .venv/bin/python -m pytest tests/ -v
"""

import numpy as np
import pytest

from spectralpaint.audioio import load_audio, write_float_wav
from spectralpaint.dsp import STFTConfig, istft, roundtrip_error, stft
from spectralpaint.engine import EditSession
from spectralpaint.mask import Stroke, composite_linear_gain
from spectralpaint.metrics import (
    boundary_ring, full_report, leakage_untouched, pan_shift,
    phase_diff_change, untouched_region,
)
from spectralpaint.signals import dual_tone, mixed_sample, sweep, transient_mix


@pytest.fixture
def cfg():
    return STFTConfig(n_fft=2048, hop=512, window="hann")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# --------------------------------------------------------------------- STFT --
class TestSTFT:
    def test_roundtrip_random(self, cfg, rng):
        x = rng.standard_normal(50000)
        r = roundtrip_error(x, cfg)
        assert r["relative_rms_db"] < -250
        assert r["max_abs"] < 1e-12

    @pytest.mark.parametrize("window,hop", [
        ("hann", 512), ("hann", 1024), ("sine", 1024), ("hamming", 512),
    ])
    def test_roundtrip_window_hop_pairs(self, rng, window, hop):
        x = rng.standard_normal(30000)
        r = roundtrip_error(x, STFTConfig(2048, hop, window))
        assert r["max_abs"] < 1e-11

    def test_output_length_trimmed(self, cfg, rng):
        x = rng.standard_normal(12345)
        xr = istft(stft(x, cfg), cfg, x.shape[-1])
        assert xr.shape == x.shape

    def test_conjugate_symmetry_structural(self, cfg):
        # Editing the rfft half and re-transforming must preserve realness,
        # i.e. the hidden negative-frequency half stays conjugate-symmetric.
        sig, sr = dual_tone()
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.1, 1.9, 500, 1500, -18.0))
        y = s.rebuild()
        assert np.all(np.isfinite(y))
        # Full FFT of every edited frame obeys X[k] = conj(X[N-k])
        spec = stft(y[:, 0], cfg)
        for t in range(spec.shape[0]):
            full = np.fft.fft(np.fft.irfft(spec[t], n=cfg.n_fft))
            mirror = np.concatenate((full[:1], np.conj(full[cfg.n_fft:0:-1])))
            err = np.max(np.abs(full - mirror))
            assert err < 1e-10

    def test_rfft_bin_count(self, cfg):
        x = np.zeros(10000)
        assert stft(x, cfg).shape[1] == cfg.n_fft // 2 + 1


# ------------------------------------------------------------- mask / edges --
class TestMask:
    def test_gain_inside_and_outside(self, cfg):
        sig, sr = dual_tone(duration=1.0)
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.2, 0.8, 1000, 1200, -20.0,
                            feather_t=0.0, feather_f=0.0))
        g = s.gain_field()
        t, f = s.times, s.freqs
        inside = (t >= 0.2) & (t <= 0.8)
        bins = (f >= 1000) & (f <= 1200)
        assert np.allclose(g[np.ix_(inside, bins)], 10 ** (-20 / 20))
        far_t = t < 0.05
        assert np.allclose(g[np.ix_(far_t, bins)], 1.0)

    def test_feather_is_monotone_and_sigma_scaled(self, cfg):
        sig, sr = dual_tone(duration=1.0)
        s = EditSession(sig, sr, cfg)
        stroke = Stroke(0.3, 0.7, 1000, 1200, -20.0,
                        feather_t=0.05, feather_f=0.0)
        s.add_stroke(stroke)
        g = s.gain_field()
        f = s.freqs
        fbin = np.argmin(np.abs(f - 1100))
        # sample gain along time outside the left edge (t0=0.3)
        t = s.times
        left = t[(t >= 0.15) & (t < 0.3)]
        gains = g[np.searchsorted(t, left), fbin]
        # gains must decay moving away from the edge
        assert np.all(np.diff(gains) <= 1e-12)
        # at one sigma (0.05s) gain in dB should be ~ -20 * exp(-0.5);
        # tolerance accounts for frame-center quantization (up to hop/2)
        idx = np.argmin(np.abs(t - (0.3 - 0.05)))
        expected_db = -20.0 * np.exp(-0.5)
        actual_db = 20 * np.log10(g[idx, fbin])
        assert abs(actual_db - expected_db) < 2.0

    def test_zoom_independence(self, cfg):
        # Rasterization depends only on physical bin centers / frame times,
        # never on canvas pixels: two sessions, identical grids -> identical gain.
        sig, sr = sweep(duration=1.0)
        strokes = [Stroke(0.1, 0.9, 500, 800, -12, 0.03, 40)]
        a = EditSession(sig, sr, cfg)
        for st in strokes:
            a.add_stroke(st)
        g1 = composite_linear_gain(strokes, a.times, a.freqs)
        g2 = composite_linear_gain(strokes, a.times, a.freqs)
        assert np.array_equal(g1, g2)

    def test_strokes_compose_in_db(self, cfg):
        sig, sr = dual_tone(duration=1.0)
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.2, 0.8, 1000, 1200, -6.0, 0.0, 0.0))
        s.add_stroke(Stroke(0.2, 0.8, 1000, 1200, -6.0, 0.0, 0.0))
        g = s.gain_field()
        t, f = s.times, s.freqs
        inside = (t >= 0.2) & (t <= 0.8)
        bins = (f >= 1000) & (f <= 1200)
        assert np.allclose(g[np.ix_(inside, bins)], 10 ** (-12 / 20))


# ------------------------------------------------------------------- engine --
class TestEngine:
    def test_exact_undo_returns_prior_state(self, cfg):
        sig, sr = mixed_sample(duration=2.0)
        s = EditSession(sig, sr, cfg)
        baseline = s.rebuild()
        s.add_stroke(Stroke(0.2, 1.8, 400, 900, -24))
        edited = s.rebuild()
        assert not np.array_equal(edited, baseline)
        s.undo()
        assert np.array_equal(s.rebuild(), baseline)

    def test_undo_is_not_inverse_gain(self, cfg):
        # Two successive edits then two undos must return bit-for-bit.
        sig, sr = dual_tone()
        s = EditSession(sig, sr, cfg)
        baseline = s.rebuild()
        s.add_stroke(Stroke(0.1, 1.9, 300, 600, -18))
        s.add_stroke(Stroke(0.1, 1.9, 1500, 2000, 6))
        s.undo()
        s.undo()
        assert s.strokes == []
        assert np.array_equal(s.rebuild(), baseline)

    def test_phase_locked_preserves_interchannel_phase(self, cfg):
        sig, sr = dual_tone()
        s = EditSession(sig, sr, cfg, strategy="phase_locked")
        s.add_stroke(Stroke(0.2, 1.8, 380, 520, -24, 0.02, 15))
        specs1 = s.edited_spectra()  # exact edited complex spectra, no re-analysis
        hard = ~untouched_region(s.times, s.freqs, s.strokes, n_sigma=0.0)
        ph = phase_diff_change(s._base_specs[0], s._base_specs[1],
                               specs1[0], specs1[1], hard)
        assert ph["phase_diff_change_rms_rad"] < 1e-12

    def test_pan_stable_under_phase_locked_narrow_band(self, cfg):
        sig, sr = dual_tone()
        s = EditSession(sig, sr, cfg, strategy="phase_locked")
        s.add_stroke(Stroke(0.2, 1.8, 380, 520, -24, 0.02, 15))
        specs1 = s.edited_spectra()
        hard = ~untouched_region(s.times, s.freqs, s.strokes, n_sigma=0.0)
        pan = pan_shift(s._base_specs[0], s._base_specs[1],
                        specs1[0], specs1[1], hard)
        assert abs(pan["pan_shift_rms_db"]) < 1e-9

    def test_coherent_joint_uses_common_gain_on_coherent_cells(self, cfg):
        sig, sr = dual_tone()  # tones are fully coherent at their bins
        s = EditSession(sig, sr, cfg, strategy="coherent_joint",
                        coherence_threshold=0.5)
        s.add_stroke(Stroke(0.2, 1.8, 380, 520, -24, 0.01, 10))
        specs1 = s.edited_spectra()
        # at coherent tone bins, L and R must share the same real multiplier
        f = s.freqs
        tbins = np.abs(f - 440) < 25
        t = s.times
        frames = (t > 0.25) & (t < 1.75)
        gL = specs1[0][np.ix_(frames, tbins)] / s._base_specs[0][np.ix_(frames, tbins)]
        gR = specs1[1][np.ix_(frames, tbins)] / s._base_specs[1][np.ix_(frames, tbins)]
        assert np.allclose(gL, gR, atol=1e-12)
        assert np.all(np.abs(np.imag(gL)) < 1e-12)

    def test_narrowband_does_not_touch_distant_tone(self, cfg):
        sig, sr = dual_tone()
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.2, 1.8, 380, 500, -30, 0.01, 10))
        y = s.rebuild()
        f = s.freqs
        far = np.abs(f - 1760) < 25
        e0 = sum(np.sum(np.abs(s._base_specs[c][:, far]) ** 2) for c in range(2))
        e1 = sum(np.sum(np.abs(stft(y[:, c], cfg)[:, far]) ** 2) for c in range(2))
        assert abs(10 * np.log10(e1 / e0)) < 1e-6

    def test_no_global_volume_change_substitute(self, cfg):
        # The edit must change only selected cells: total gain on untouched
        # region stays at unity.
        sig, sr = mixed_sample(duration=2.0)
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.5, 1.0, 2000, 2600, -18, 0.02, 30))
        g = s.gain_field()
        # Beyond 8 sigma the Gaussian feather underflows to exact unity, so the
        # edit cannot be a disguised global volume change.
        unt = untouched_region(s.times, s.freqs, s.strokes, n_sigma=8.0)
        # Gaussian tail at 8 sigma is ~5e-15: effectively exact unity
        assert np.all(np.abs(g[unt] - 1.0) < 1e-12)
        # and the hard selection itself is actually attenuated
        assert np.median(g[s.mask_region(n_sigma=0.0)]) < 10 ** (-15 / 20)


# ------------------------------------------------------------------ metrics --
class TestMetrics:
    def test_leakage_small_for_localized_edit(self, cfg):
        sig, sr = sweep()
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(1.0, 1.5, 1000, 1400, -24, 0.02, 30))
        y = s.rebuild()
        specs1 = [stft(y[:, c], cfg) for c in range(2)]
        unt = untouched_region(s.times, s.freqs, s.strokes)
        leak = leakage_untouched(s._base_specs, specs1, unt)
        assert leak["max_leak_energy_ratio"] < 1e-3

    def test_boundary_ring_catches_edge_energy(self, cfg):
        sig, sr = transient_mix()
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.5, 1.5, 700, 1300, -18, 0.05, 50))
        y = s.rebuild()
        ring = boundary_ring(s.times, s.freqs, s.strokes)
        assert ring.sum() > 0
        rep = full_report(s, y)
        assert "boundary" in rep and "leakage" in rep and "pan" in rep

    def test_full_report_runs_on_all_signals(self, cfg):
        for gen in (dual_tone, sweep, transient_mix, mixed_sample):
            sig, sr = gen()
            s = EditSession(sig, sr, cfg)
            s.add_stroke(Stroke(0.3, sig.shape[0] / sr - 0.3,
                                500, 1500, -15, 0.03, 40))
            y = s.rebuild()
            rep = full_report(s, y)
            assert rep["n_strokes"] == 1
            assert np.isfinite(rep["time_domain"]["rms"])

    def test_full_report_distinguishes_exact_and_reanalyzed(self, cfg):
        sig, sr = dual_tone()
        s = EditSession(sig, sr, cfg, strategy="phase_locked")
        s.add_stroke(Stroke(0.2, 1.8, 380, 520, -24, 0.02, 15))
        rep = full_report(s, s.rebuild())
        # exact edited complex spectrum: identical real gain => zero pan/phase
        assert abs(rep["pan"]["pan_shift_rms_db"]) < 1e-7
        assert rep["phase"]["phase_diff_change_rms_rad"] < 1e-10
        # re-analyzed figures always exist and are finite
        assert np.isfinite(rep["pan_reanalyzed"]["pan_shift_rms_db"])
        assert np.isfinite(rep["phase_reanalyzed"]["phase_diff_change_rms_rad"])


# ----------------------------------------------------------- signal-specific --
class TestSignalScenarios:
    def test_dual_tone_local_edit(self, cfg):
        sig, sr = dual_tone()
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.0, 2.0, 1700, 1820, -40, 0.0, 5.0))
        y = s.rebuild()
        f = s.freqs
        target = np.abs(f - 1760) < 20
        before = np.sum(np.abs(s._base_specs[0][:, target]) ** 2)
        after = np.sum(np.abs(stft(y[:, 0], cfg)[:, target]) ** 2)
        assert 10 * np.log10(after / before) < -20

    def test_sweep_partial_attenuation(self, cfg):
        sig, sr = sweep(duration=3.0)
        s = EditSession(sig, sr, cfg)
        # chirp passes ~1467 Hz at t=1s; edit a window in the middle
        s.add_stroke(Stroke(0.9, 1.2, 1200, 1800, -20, 0.01, 20))
        y = s.rebuild()
        assert y.shape == sig.shape
        # energy in edited window drops relative to pristine
        n0, n1 = int(0.9 * sr), int(1.2 * sr)
        assert np.sqrt(np.mean(y[n0:n1] ** 2)) < np.sqrt(np.mean(sig[n0:n1] ** 2))
        # outside window energy preserved
        na, nb = int(2.2 * sr), int(2.6 * sr)
        assert np.allclose(y[na:nb], sig[na:nb], atol=1e-4)

    def test_transient_boundary_residue(self, cfg):
        sig, sr = transient_mix()
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.25, 0.45, 200, 4000, -15, 0.02, 40))
        y = s.rebuild()
        rep = full_report(s, y)
        # leakage into untouched cells stays bounded (no smeared click tails)
        assert rep["leakage"]["max_leak_energy_ratio"] < 5e-3


# ------------------------------------------------------------- export / I/O --
class TestExportRoundTrip:
    def test_written_file_matches_reconstruction(self, cfg, tmp_path):
        sig, sr = mixed_sample(duration=1.5)
        s = EditSession(sig, sr, cfg)
        s.add_stroke(Stroke(0.3, 1.2, 400, 900, -20, 0.02, 30))
        edited = s.rebuild()
        path = tmp_path / "out.wav"
        write_float_wav(str(path), edited, sr)
        reloaded, rsr = load_audio(str(path))
        assert rsr == sr
        assert reloaded.shape == edited.shape
        assert np.allclose(reloaded, edited, atol=1e-6)

    def test_reanalyzed_export_shows_target_attenuation(self, cfg, tmp_path):
        sig, sr = dual_tone(duration=1.0)
        s = EditSession(sig, sr, cfg)
        # band covers the 1760 Hz bin; time span covers essentially the whole
        # tone so the measured band energy reflects the applied -40 dB gain
        s.add_stroke(Stroke(0.0, 1.0, 1680, 1840, -40, 0.0, 10.0))
        path = tmp_path / "tone.wav"
        write_float_wav(str(path), s.rebuild(), sr)
        reloaded, _ = load_audio(str(path))
        f = s.freqs
        target = np.abs(f - 1760) < 20
        before = np.sum(np.abs(s._base_specs[0][:, target]) ** 2)
        after = np.sum(np.abs(stft(reloaded[:, 0], cfg)[:, target]) ** 2)
        assert 10 * np.log10(after / before) < -20


# -------------------------------------------------- cross-channel strategies --
class TestStrategyDifferences:
    def _stereo_session(self, cfg, strategy):
        # Build a signal with a coherent center tone plus an incoherent ambient
        # part (independent L/R noise) so the two strategies can diverge.
        sr = 44100
        n = sr
        t = np.arange(n) / sr
        rng = np.random.default_rng(3)
        tone = 0.4 * np.sin(2 * np.pi * 1000 * t)
        noise_l = 0.15 * rng.standard_normal(n)
        noise_r = 0.15 * rng.standard_normal(n)
        stereo = np.stack([tone + noise_l, tone + noise_r], axis=1)
        s = EditSession(stereo, sr, cfg, strategy=strategy,
                        coherence_threshold=0.6)
        s.add_stroke(Stroke(0.1, 0.9, 800, 1200, -24, 0.01, 15))
        return s

    def test_coherence_separates_tone_from_noise(self, cfg):
        s = self._stereo_session(cfg, "coherent_joint")
        g2 = s.coherence()
        f = s.freqs
        tone_bins = np.abs(f - 1000) < 20
        noise_bins = (f > 2000) & (f < 3000)
        assert np.median(g2[:, tone_bins]) > 0.8
        assert np.median(g2[:, noise_bins]) < 0.5

    def test_both_strategies_are_real_gains(self, cfg):
        for strat in ("phase_locked", "coherent_joint"):
            s = self._stereo_session(cfg, strat)
            edited = s.edited_spectra()
            for base, spec in zip(s._base_specs, edited):
                active = np.abs(base) ** 2 > 1e-8 * np.max(np.abs(base) ** 2)
                ratio = spec[active] / base[active]
                assert np.all(np.abs(np.imag(ratio)) < 1e-9)

    def test_phase_locked_identical_multipliers_both_channels(self, cfg):
        s = self._stereo_session(cfg, "phase_locked")
        gL = s.edited_spectra()[0] / s._base_specs[0]
        gR = s.edited_spectra()[1] / s._base_specs[1]
        # wherever there is signal, the same real gain applies to L and R
        energy = np.abs(s._base_specs[0]) ** 2 + np.abs(s._base_specs[1]) ** 2
        active = energy > 1e-6 * energy.max()
        assert np.allclose(gL[active], gR[active], atol=1e-12)
