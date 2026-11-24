"""Tests for the selection storage representations (谱选区表示实验).

Covers:
  * curve/rect equivalence with the engine's Stroke rasterization;
  * window-change behavior of curve / mask_pixel / mask_resample / mask_strict;
  * adjacent-band impact on the shared narrowband tone after a window change;
  * transient time-boundary behavior;
  * undo cost (persistent size) and exact undo after a window change;
  * the production endpoint /api/reconfigure (selections survive re-gridding,
    kernel failures come back as JSON errors).
"""

import json
import threading
import urllib.request

import numpy as np
import pytest

from spectralpaint.dsp import (STFTConfig, frame_count, frame_times,
                               )
from spectralpaint.dsp import freq_bin_centers, istft, stft
from spectralpaint.engine import EditSession
from spectralpaint.mask import Stroke, stroke_gain_db
from spectralpaint.selection import (CurveSelection, RasterizedMask,
                                     half_gain_bounds)

SR = 44100
W1 = STFTConfig(n_fft=2048, hop=512, window="hann")
W2 = STFTConfig(n_fft=4096, hop=1024, window="hann")
NOTCH = Stroke(0.4, 1.6, 940.0, 1060.0, -40.0, 0.01, 10.0)
TEDIT = Stroke(0.5, 1.0, 200.0, 4000.0, -15.0, 0.02, 40.0)
CLICKS = (0.3, 0.725, 1.15, 1.575)


def _grid(cfg, n_samples=88200, sr=SR):
    nf = frame_count(n_samples, cfg.n_fft, cfg.hop)
    return frame_times(cfg, nf, sr), freq_bin_centers(cfg, sr), nf


def _narrowband(sr=SR, dur=2.0):
    t = np.arange(int(sr * dur)) / sr
    return 0.4 * np.sin(2 * np.pi * 1000 * t) + 0.25 * np.sin(2 * np.pi * 500 * t)


def _transient(sr=SR, dur=2.0):
    n = int(sr * dur)
    t = np.arange(n) / sr
    x = 0.15 * np.sin(2 * np.pi * 1000 * t)
    for ct in CLICKS:
        i = int(ct * sr)
        L = int(0.008 * sr)
        x[i:i + L] += 0.8 * np.exp(-np.linspace(0, 6, L))
    return x


def _apply(x, cfg, field_db):
    return istft(stft(x, cfg) * 10.0 ** (field_db / 20.0), cfg, x.shape[0])


def _band_change_db(x0, x1, cfg, f_lo, f_hi, t_lo, t_hi, sr=SR):
    s0, s1 = stft(x0, cfg), stft(x1, cfg)
    times, freqs, _ = _grid(cfg, x0.shape[0], sr)
    ix = np.ix_((times >= t_lo) & (times <= t_hi),
                (freqs >= f_lo) & (freqs <= f_hi))
    e0 = np.sum(np.abs(s0[ix]) ** 2)
    e1 = np.sum(np.abs(s1[ix]) ** 2)
    return 10 * np.log10((e1 + 1e-30) / (e0 + 1e-30))


def _click_change_db(x0, x1, ct, sr=SR, half=0.04):
    i0, i1 = int((ct - half) * sr), int((ct + half) * sr)
    r0 = np.sqrt(np.mean(x0[i0:i1] ** 2))
    r1 = np.sqrt(np.mean(x1[i0:i1] ** 2))
    return 20 * np.log10((r1 + 1e-30) / (r0 + 1e-30))


# ------------------------------------------------------------------ basics --
class TestCurveSelection:
    def test_rect_curve_matches_stroke_bitexact(self):
        times, freqs, _ = _grid(W1)
        s = NOTCH.normalized()
        curve = CurveSelection.from_stroke(s)
        a = curve.gain_db_field(times, freqs)
        b = stroke_gain_db(s, times, freqs)
        assert np.array_equal(a, b)

    def test_nonrect_curve_follows_boundary(self):
        # V-shaped band: f_lo rises linearly from 400 to 800 Hz
        curve = CurveSelection((0.2, 1.8), (400.0, 800.0), (600.0, 1000.0),
                               -12.0, 0.0, 0.0)
        # evaluate exactly at the control times to avoid frame-center rounding
        times = np.array([0.2, 1.0, 1.8])
        freqs = np.array([300.0, 500.0, 700.0, 900.0, 1100.0])
        g = curve.gain_db_field(times, freqs)
        # t=0.2: band 400-600 -> 500 in, 700/300 out
        assert g[0, 1] == -12.0 and g[0, 3] == 0.0 and g[0, 0] == 0.0
        # t=1.0: band 600-800 -> 700 in, 500/900 out
        assert g[1, 2] == -12.0 and g[1, 1] == 0.0 and g[1, 3] == 0.0
        # t=1.8: band 800-1000 -> 900 in, 500 out
        assert g[2, 3] == -12.0 and g[2, 1] == 0.0

    def test_invalid_curve_rejected(self):
        with pytest.raises(ValueError):
            CurveSelection((0.0, 1.0), (900.0, 900.0), (100.0, 100.0), -6.0)


# ------------------------------------------------------------ window change --
class TestWindowChange:
    def test_curve_boundary_stable_across_windows(self):
        t1, f1, _ = _grid(W1)
        t2, f2, _ = _grid(W2)
        curve = CurveSelection.from_stroke(NOTCH)
        b1 = half_gain_bounds(curve.gain_db_field(t1, f1), t1, f1, NOTCH.gain_db)
        b2 = half_gain_bounds(curve.gain_db_field(t2, f2), t2, f2, NOTCH.gain_db)
        # drift bounded by the new grid's quantization step
        assert abs(b2["f_lo"] - b1["f_lo"]) <= (SR / W2.n_fft) + 1e-9
        assert abs(b2["f_hi"] - b1["f_hi"]) <= (SR / W2.n_fft) + 1e-9
        assert abs(b2["t_lo"] - b1["t_lo"]) <= (W2.hop / SR) + 1e-9

    def test_pixel_mask_shifts_band_with_window(self):
        t1, f1, n1 = _grid(W1)
        t2, f2, n2 = _grid(W2)
        mask = RasterizedMask.from_strokes([NOTCH], W1, SR, n1)
        b1 = half_gain_bounds(mask.gain_db_field(W1, n1, "strict"), t1, f1,
                              NOTCH.gain_db)
        field2 = mask.gain_db_field(W2, n2, "pixel")
        b2 = half_gain_bounds(field2, t2, f2, NOTCH.gain_db)
        # bin width halves: the frozen band lands near half its frequency
        assert b1["f_lo"] == pytest.approx(947.5, abs=2.0)
        assert b2["f_lo"] == pytest.approx(b1["f_lo"] / 2, abs=SR / W2.n_fft)
        assert b2["f_hi"] < 600.0  # nowhere near the intended 1060 Hz

    def test_strict_mask_refuses_other_window(self):
        _, _, n1 = _grid(W1)
        _, _, n2 = _grid(W2)
        mask = RasterizedMask.from_strokes([NOTCH], W1, SR, n1)
        with pytest.raises(ValueError, match="bound to its original window"):
            mask.gain_db_field(W2, n2, "strict")
        # same window re-opens fine
        assert mask.gain_db_field(W1, n1, "strict").shape == mask.gain_db.shape

    def test_resample_keeps_band_within_old_quantization(self):
        t1, f1, n1 = _grid(W1)
        t2, f2, n2 = _grid(W2)
        mask = RasterizedMask.from_strokes([NOTCH], W1, SR, n1)
        b1 = half_gain_bounds(mask.gain_db_field(W1, n1, "strict"), t1, f1,
                              NOTCH.gain_db)
        b2 = half_gain_bounds(mask.gain_db_field(W2, n2, "resample"), t2, f2,
                              NOTCH.gain_db)
        old_bin = SR / W1.n_fft
        assert abs(b2["f_lo"] - b1["f_lo"]) <= old_bin
        assert abs(b2["f_hi"] - b1["f_hi"]) <= old_bin


# ------------------------------------------------- narrowband adjacent band --
class TestNarrowbandAfterWindowChange:
    def test_same_tone_adjacent_band(self):
        x = _narrowband()
        n = x.shape[0]
        _, _, n2 = _grid(W2, n)
        t2, f2, _ = _grid(W2, n)
        curve = CurveSelection.from_stroke(NOTCH)
        mask = RasterizedMask.from_strokes(
            [NOTCH], W1, SR, frame_count(n, W1.n_fft, W1.hop))

        edits = {
            "curve": _apply(x, W2, curve.gain_db_field(t2, f2)),
            "mask_pixel": _apply(x, W2, mask.gain_db_field(W2, n2, "pixel")),
            "mask_resample": _apply(x, W2, mask.gain_db_field(W2, n2, "resample")),
        }
        att = {k: _band_change_db(x, y, W2, 985, 1015, 0.5, 1.5)
               for k, y in edits.items()}
        adj = {k: _band_change_db(x, y, W2, 485, 515, 0.5, 1.5)
               for k, y in edits.items()}
        # curve: intent achieved, adjacent band untouched
        assert att["curve"] < -30.0
        assert abs(adj["curve"]) < 0.01
        # pixel-fixed: intent lost, adjacent band damaged
        assert att["mask_pixel"] > -0.01
        assert adj["mask_pixel"] < -3.0
        # resample: band right, adjacent untouched
        assert att["mask_resample"] < -30.0
        assert abs(adj["mask_resample"]) < 0.01


# ---------------------------------------------------------------- transient --
class TestTransientAfterWindowChange:
    def test_click_selectivity(self):
        x = _transient()
        n = x.shape[0]
        _, _, n2 = _grid(W2, n)
        t2, f2, _ = _grid(W2, n)
        curve = CurveSelection.from_stroke(TEDIT)
        mask = RasterizedMask.from_strokes(
            [TEDIT], W1, SR, frame_count(n, W1.n_fft, W1.hop))
        y_curve = _apply(x, W2, curve.gain_db_field(t2, f2))
        y_pixel = _apply(x, W2, mask.gain_db_field(W2, n2, "pixel"))

        c_curve = [_click_change_db(x, y_curve, ct) for ct in CLICKS]
        c_pixel = [_click_change_db(x, y_pixel, ct) for ct in CLICKS]
        # curve: only click@0.725 is attenuated
        assert c_curve[1] < -5.0
        assert all(abs(c_curve[i]) < 0.5 for i in (0, 2, 3))
        # pixel-fixed: click@0.725 missed, later clicks wrongly attenuated
        assert abs(c_pixel[1]) < 0.5
        assert c_pixel[2] < -5.0 and c_pixel[3] < -5.0


# ---------------------------------------------------------------- undo cost --
class TestUndoCost:
    def test_storage_size_orders_of_magnitude(self):
        _, _, n1 = _grid(W1)
        curve = CurveSelection.from_stroke(NOTCH)
        mask = RasterizedMask.from_strokes([NOTCH], W1, SR, n1)
        assert curve.storage_bytes < 1024
        assert mask.storage_bytes > 100 * curve.storage_bytes

    def test_exact_undo_after_window_reconfigure(self):
        # Selections stored as continuous boundaries survive a window change;
        # undo on the new grid still returns bit-exactly to the prior state.
        x = _narrowband()
        s1 = EditSession(x[:, None], SR, W1)
        s1.add_stroke(Stroke(0.4, 1.6, 940.0, 1060.0, -40.0, 0.01, 10.0))
        strokes = [Stroke(**vars(st)) for st in s1.strokes]
        s2 = EditSession(x[:, None], SR, W2)
        s2.strokes = strokes
        baseline = s2.rebuild()
        s2.add_stroke(Stroke(0.1, 1.9, 2000.0, 3000.0, -6.0))
        assert not np.array_equal(s2.rebuild(), baseline)
        s2.undo()
        assert np.array_equal(s2.rebuild(), baseline)


# ------------------------------------------------------- production endpoint --
class TestReconfigureEndpoint:
    @pytest.fixture
    def server(self):
        from spectralpaint import server as srv
        httpd = srv.serve("127.0.0.1", 0)
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        yield f"http://127.0.0.1:{port}"
        httpd.shutdown()
        th.join(timeout=5)

    @staticmethod
    def _post(base, path, body):
        req = urllib.request.Request(
            base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_window_change_keeps_selections(self, server):
        st, j = self._post(server, "/api/session/demo",
                           {"demo": "dual_tone", "n_fft": 2048, "hop": 512})
        assert st == 200 and j["n_bins"] == 1025
        st, j = self._post(server, "/api/edit",
                           {"t0": 0.2, "t1": 1.8, "f0": 380, "f1": 500,
                            "gain_db": -24.0})
        assert st == 200 and len(j["strokes"]) == 1
        st, j = self._post(server, "/api/reconfigure",
                           {"n_fft": 4096, "hop": 1024, "window": "hann"})
        assert st == 200
        assert j["n_bins"] == 2049 and j["reconfigured"]["kept_strokes"] == 1
        assert len(j["strokes"]) == 1
        # the stroke is still the same physical selection (s / Hz)
        assert j["strokes"][0]["f0"] == pytest.approx(380.0)
        assert j["strokes"][0]["f1"] == pytest.approx(500.0)
        # rebuild + audio + export all work on the new grid
        st, j = self._post(server, "/api/rebuild", {})
        assert st == 200 and "report" in j
        st, j = self._post(server, "/api/export", {"name": "reconf_test.wav"})
        assert st == 200 and j["export"]["matches_in_memory"]
        with urllib.request.urlopen(server + "/api/audio?which=edited") as r:
            assert r.status == 200 and len(r.read()) > 1000

    def test_kernel_failure_reported_as_json(self, server):
        st, j = self._post(server, "/api/session/demo", {"demo": "dual_tone"})
        assert st == 200
        # hop > n_fft is rejected by the DSP layer; the UI shows this message
        st, j = self._post(server, "/api/reconfigure",
                           {"n_fft": 1024, "hop": 4096, "window": "hann"})
        assert st == 400 and "error" in j
