"""Spectral selection representation experiment (谱选区表示实验).

Compares two selection *storage* representations on the SAME inverse-transform
prototype (:func:`spectralpaint.dsp.istft`):

  * ``curve``        -- continuous t/f boundary curves (seconds / Hz),
                        re-evaluated on whatever STFT grid is current;
  * ``mask_pixel``   -- coefficient mask stored with its original window
                        parameters, re-opened index-for-index ("pixel-fixed");
  * ``mask_resample``-- the same stored mask, linearly resampled in physical
                        coordinates onto the new grid;
  * ``mask_strict``  -- the same stored mask, refusing any grid but its own.

Protocol
--------
1. Build the edit at window W1 (n_fft=2048, hop=512) and store it as a curve
   and as a coefficient mask.
2. Change the analysis window to W2 (n_fft=4096, hop=1024) and re-open the
   stored selections under each policy.
3. Measure, on the same narrowband tone pair and the same transient material:
     * mask boundary drift: where the half-gain contour lands vs. where it
       landed at W1 (Hz and seconds);
     * reconstruction error: notch depth at the target tone, energy change in
       the adjacent band, per-click level changes for transients;
     * undo cost: persistent bytes per selection and per-rebuild evaluation
       time of each representation.

Outputs (all regenerated on every run; docs/selection_representation.md is
rendered from these numbers, never hand-edited):
  exports/selection_experiment/results.json
  exports/selection_experiment/*.wav
  exports/selection_experiment/figures/*.png
  docs/selection_representation.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectralpaint.audioio import write_float_wav
from spectralpaint.dsp import (STFTConfig, frame_count, frame_times,
                               freq_bin_centers, istft, stft)
from spectralpaint.mask import Stroke
from spectralpaint.selection import (CurveSelection, RasterizedMask,
                                     half_gain_bounds)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exports" / "selection_experiment"
FIG = OUT / "figures"
DOC = ROOT / "docs" / "selection_representation.md"

SR = 44100
W1 = STFTConfig(n_fft=2048, hop=512, window="hann")
W2 = STFTConfig(n_fft=4096, hop=1024, window="hann")

NOTCH = Stroke(t0=0.4, t1=1.6, f0=940.0, f1=1060.0, gain_db=-40.0,
               feather_t=0.01, feather_f=10.0)
TRANSIENT_EDIT = Stroke(t0=0.5, t1=1.0, f0=200.0, f1=4000.0, gain_db=-15.0,
                        feather_t=0.02, feather_f=40.0)
CLICK_TIMES = (0.3, 0.725, 1.15, 1.575)


# ----------------------------------------------------------------- signals --
def narrowband_pair(sr: int = SR, duration: float = 2.0) -> np.ndarray:
    """Two steady tones: 1000 Hz (notch target) and 500 Hz (adjacent band)."""
    t = np.arange(int(sr * duration)) / sr
    return 0.4 * np.sin(2 * np.pi * 1000.0 * t) + 0.25 * np.sin(2 * np.pi * 500.0 * t)


def transient_clicks(sr: int = SR, duration: float = 2.0) -> np.ndarray:
    """Low tone plus four 8 ms clicks at known, well-separated times."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    x = 0.15 * np.sin(2 * np.pi * 1000.0 * t)
    for ct in CLICK_TIMES:
        i = int(ct * sr)
        length = int(0.008 * sr)
        env = np.exp(-np.linspace(0, 6, length))
        x[i:i + length] += 0.8 * env
    return x


# ------------------------------------------------------------------ helpers --
def apply_gain_db(x: np.ndarray, cfg: STFTConfig, field_db: np.ndarray) -> np.ndarray:
    """The shared prototype: rfft STFT -> real gain on complex bins -> ISTFT/OLA."""
    spec = stft(x, cfg)
    return istft(spec * 10.0 ** (field_db / 20.0), cfg, x.shape[0])


def grid(cfg: STFTConfig, n_samples: int):
    n_frames = frame_count(n_samples, cfg.n_fft, cfg.hop)
    return frame_times(cfg, n_frames, SR), freq_bin_centers(cfg, SR), n_frames


def band_energy_change_db(x0, x1, cfg, f_lo, f_hi, t_lo, t_hi) -> float:
    """Re-analyzed energy change (dB) inside a physical t/f region."""
    s0, s1 = stft(x0, cfg), stft(x1, cfg)
    times, freqs, _ = grid(cfg, x0.shape[0])
    sel_t = (times >= t_lo) & (times <= t_hi)
    sel_f = (freqs >= f_lo) & (freqs <= f_hi)
    ix = np.ix_(sel_t, sel_f)
    e0 = np.sum(np.abs(s0[ix]) ** 2)
    e1 = np.sum(np.abs(s1[ix]) ** 2)
    return float(10.0 * np.log10((e1 + 1e-30) / (e0 + 1e-30)))


def click_level_change_db(x0, x1, t_click, half: float = 0.04) -> float:
    i0, i1 = int((t_click - half) * SR), int((t_click + half) * SR)
    r0 = np.sqrt(np.mean(x0[i0:i1] ** 2))
    r1 = np.sqrt(np.mean(x1[i0:i1] ** 2))
    return float(20.0 * np.log10((r1 + 1e-30) / (r0 + 1e-30)))


def variant_fields(intent: Stroke, n_samples: int, cfg: STFTConfig,
                   mask: RasterizedMask) -> dict:
    """Evaluate every representation of ``intent`` on ``cfg``'s grid.

    ``mask`` is the coefficient mask stored at creation time (W1); re-opening
    it under a different window is the whole point of the experiment.
    """
    times, freqs, n_frames = grid(cfg, n_samples)
    curve = CurveSelection.from_stroke(intent)
    fields = {"curve": curve.gain_db_field(times, freqs)}
    fields["mask_pixel"] = mask.gain_db_field(cfg, n_frames, policy="pixel")
    fields["mask_resample"] = mask.gain_db_field(cfg, n_frames, policy="resample")
    try:
        fields["mask_strict"] = mask.gain_db_field(cfg, n_frames, policy="strict")
    except ValueError as exc:
        fields["mask_strict"] = str(exc)
    return fields


def boundary_row(field_db, times, freqs, gain_db, ref, intent: Stroke):
    b = half_gain_bounds(field_db, times, freqs, gain_db)
    if b is None:
        return {"measured": None}
    n = intent.normalized()
    row = {"measured": b}
    for k in ("f_lo", "f_hi", "t_lo", "t_hi"):
        row[f"d_{k}"] = b[k] - ref[k]            # drift vs W1 reference
    row["err_f_vs_intent"] = max(abs(b["f_lo"] - n.f0), abs(b["f_hi"] - n.f1))
    row["err_t_vs_intent"] = max(abs(b["t_lo"] - n.t0), abs(b["t_hi"] - n.t1))
    return row


# ------------------------------------------------------------- experiments --
def run_narrowband() -> dict:
    x = narrowband_pair()
    n = x.shape[0]
    res: dict = {"intent": vars(NOTCH), "windows": {"W1": vars(W1), "W2": vars(W2)}}

    # reference: continuous curve evaluated at the creation window W1
    t1, f1, _ = grid(W1, n)
    ref_field = CurveSelection.from_stroke(NOTCH).gain_db_field(t1, f1)
    ref = half_gain_bounds(ref_field, t1, f1, NOTCH.gain_db)
    res["reference_W1"] = {"bounds": ref}
    ref_edit = apply_gain_db(x, W1, ref_field)
    res["reference_W1"]["att_1000_db"] = band_energy_change_db(
        x, ref_edit, W1, 985, 1015, 0.5, 1.5)
    res["reference_W1"]["adj_500_db"] = band_energy_change_db(
        x, ref_edit, W1, 485, 515, 0.5, 1.5)

    # reopen at W2 under every representation
    t2, f2, n2 = grid(W2, n)
    mask = RasterizedMask.from_strokes(
        [NOTCH], W1, SR, frame_count(n, W1.n_fft, W1.hop))
    fields = variant_fields(NOTCH, n, W2, mask)
    res["W2"] = {}
    for name, field in fields.items():
        if isinstance(field, str):  # strict policy refused
            res["W2"][name] = {"error": field}
            continue
        row = boundary_row(field, t2, f2, NOTCH.gain_db, ref, NOTCH)
        edited = apply_gain_db(x, W2, field)
        row["att_1000_db"] = band_energy_change_db(x, edited, W2, 985, 1015, 0.5, 1.5)
        row["adj_500_db"] = band_energy_change_db(x, edited, W2, 485, 515, 0.5, 1.5)
        write_float_wav(str(OUT / f"narrowband_w2_{name}.wav"), edited, SR)
        res["W2"][name] = row
    write_float_wav(str(OUT / "narrowband_pristine.wav"), x, SR)
    res["signal"] = "tones 1000 Hz (0.4) + 500 Hz (0.25), mono, 2.0 s"
    return res, x, fields


def run_transient() -> dict:
    x = transient_clicks()
    n = x.shape[0]
    res: dict = {"intent": vars(TRANSIENT_EDIT), "click_times": list(CLICK_TIMES)}

    t1, f1, _ = grid(W1, n)
    ref_field = CurveSelection.from_stroke(TRANSIENT_EDIT).gain_db_field(t1, f1)
    ref = half_gain_bounds(ref_field, t1, f1, TRANSIENT_EDIT.gain_db)
    res["reference_W1"] = {"bounds": ref}
    ref_edit = apply_gain_db(x, W1, ref_field)
    res["reference_W1"]["clicks_db"] = [
        click_level_change_db(x, ref_edit, ct) for ct in CLICK_TIMES]

    t2, f2, n2 = grid(W2, n)
    mask = RasterizedMask.from_strokes(
        [TRANSIENT_EDIT], W1, SR, frame_count(n, W1.n_fft, W1.hop))
    fields = variant_fields(TRANSIENT_EDIT, n, W2, mask)
    res["W2"] = {}
    for name, field in fields.items():
        if isinstance(field, str):
            res["W2"][name] = {"error": field}
            continue
        row = boundary_row(field, t2, f2, TRANSIENT_EDIT.gain_db, ref,
                           TRANSIENT_EDIT)
        edited = apply_gain_db(x, W2, field)
        row["clicks_db"] = [click_level_change_db(x, edited, ct)
                            for ct in CLICK_TIMES]
        write_float_wav(str(OUT / f"transient_w2_{name}.wav"), edited, SR)
        res["W2"][name] = row
    write_float_wav(str(OUT / "transient_pristine.wav"), x, SR)
    res["signal"] = "tone 1000 Hz (0.15) + 4 clicks at " + ", ".join(
        f"{c}" for c in CLICK_TIMES) + " s"
    return res, x, fields


def run_undo_cost(n_samples: int) -> dict:
    """Storage bytes and per-rebuild evaluation cost of each representation."""
    times, freqs, n_frames = grid(W1, n_samples)
    curve = CurveSelection.from_stroke(NOTCH)
    mask = RasterizedMask.from_strokes([NOTCH], W1, SR, n_frames)

    reps = 60
    t0 = time.perf_counter()
    for _ in range(reps):
        curve.gain_db_field(times, freqs)
    curve_ms = (time.perf_counter() - t0) / reps * 1e3

    t0 = time.perf_counter()
    for _ in range(reps):
        mask.gain_db_field(W1, n_frames, policy="strict")
    mask_ms = (time.perf_counter() - t0) / reps * 1e3

    return {
        "curve_storage_bytes": curve.storage_bytes,
        "mask_storage_bytes": mask.storage_bytes,
        "mask_grid": [int(n_frames), int(W1.n_fft // 2 + 1)],
        "curve_eval_ms_per_rebuild": curve_ms,
        "mask_eval_ms_per_rebuild": mask_ms,
        "note": ("both representations undo exactly (drop-from-list + rebuild "
                 "from pristine, proven bit-exact in the test suite); the cost "
                 "difference is persistent size and per-rebuild evaluation"),
    }


# ----------------------------------------------------------------- figures --
def make_figures(x_nb, x_tr, nb_fields, tr_fields):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = x_nb.shape[0]
    t2, f2, _ = grid(W2, n)

    # 1) mask boundaries on the W2 grid (narrowband)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, name in zip(axes, ("curve", "mask_pixel", "mask_resample")):
        fld = nb_fields[name]
        extent = [t2[0], t2[-1], f2[0], f2[-1]]
        ax.imshow(fld.T, origin="lower", aspect="auto", cmap="viridis",
                  extent=extent, vmin=-40, vmax=0)
        ax.plot([NOTCH.t0, NOTCH.t1, NOTCH.t1, NOTCH.t0, NOTCH.t0],
                [NOTCH.f0, NOTCH.f0, NOTCH.f1, NOTCH.f1, NOTCH.f0],
                "w--", lw=1.2, label="intent")
        b = half_gain_bounds(fld, t2, f2, NOTCH.gain_db)
        if b:
            ax.plot([b["t_lo"], b["t_hi"], b["t_hi"], b["t_lo"], b["t_lo"]],
                    [b["f_lo"], b["f_lo"], b["f_hi"], b["f_hi"], b["f_lo"]],
                    "r-", lw=1.2, label="measured 1/2-gain")
        ax.set_ylim(200, 1200)
        ax.set_xlim(0.2, 1.8)
        ax.set_title(name)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Narrowband notch re-opened at W2 (n_fft=4096, hop=1024): "
                 "mask boundary per representation")
    fig.tight_layout()
    fig.savefig(FIG / "narrowband_mask_boundary_w2.png", dpi=110)
    plt.close(fig)

    # 2) spectra of the edited second (0.5-1.5 s) for each variant
    fig, ax = plt.subplots(figsize=(10, 4.5))
    seg = slice(int(0.5 * SR), int(1.5 * SR))
    win = np.hanning(seg.stop - seg.start)
    f_ax = np.fft.rfftfreq(seg.stop - seg.start, 1 / SR)

    def spec_curve(sig):
        return 20 * np.log10(np.abs(np.fft.rfft(sig[seg] * win)) + 1e-9)

    ax.plot(f_ax, spec_curve(x_nb), "k-", lw=1.4, label="pristine")
    for name, style in (("curve", "g-"), ("mask_pixel", "r-"),
                        ("mask_resample", "b-")):
        fld = nb_fields[name]
        edited = apply_gain_db(x_nb, W2, fld)
        ax.plot(f_ax, spec_curve(edited), style, lw=1.0, alpha=0.85, label=name)
    ax.set_xlim(300, 1200)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title("Re-analyzed spectrum 0.5-1.5 s after re-opening at W2 "
                 "(intent: -40 dB notch 940-1060 Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "narrowband_spectra_w2.png", dpi=110)
    plt.close(fig)

    # 3) transient: waveforms + per-click level change
    fig, (axw, axb) = plt.subplots(2, 1, figsize=(11, 6.5),
                                   gridspec_kw={"height_ratios": [2, 1]})
    t_ax = np.arange(n) / SR
    axw.plot(t_ax, x_tr, "k-", lw=0.6, label="pristine")
    colors = {"curve": "g", "mask_pixel": "r", "mask_resample": "b"}
    edits = {}
    for name in ("curve", "mask_pixel", "mask_resample"):
        edited = apply_gain_db(x_tr, W2, tr_fields[name])
        edits[name] = edited
        axw.plot(t_ax, edited, colors[name], lw=0.5, alpha=0.7, label=name)
    axw.axvspan(TRANSIENT_EDIT.t0, TRANSIENT_EDIT.t1, color="gray", alpha=0.2,
                label="intent span")
    for ct in CLICK_TIMES:
        axw.axvline(ct, color="orange", lw=0.6, ls=":")
    axw.set_xlabel("time (s)")
    axw.set_ylabel("amplitude")
    axw.set_title("Transient edit re-opened at W2 (intent: -15 dB over "
                  "0.5-1.0 s, click #2 only)")
    axw.legend(loc="upper right", fontsize=8)
    width = 0.25
    for k, name in enumerate(("curve", "mask_pixel", "mask_resample")):
        vals = [click_level_change_db(x_tr, edits[name], ct)
                for ct in CLICK_TIMES]
        axb.bar(np.arange(4) + (k - 1) * width, vals, width,
                color=colors[name], label=name)
    axb.set_xticks(np.arange(4))
    axb.set_xticklabels([f"click@{c}s" for c in CLICK_TIMES])
    axb.set_ylabel("level change (dB)")
    axb.axhline(0, color="k", lw=0.6)
    axb.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "transient_w2.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ report --
def render_report(nb, tr, undo) -> str:
    L = []
    A = L.append
    A("# 谱选区表示实验（selection representation experiment）")
    A("")
    A("本文件由 `scripts/run_selection_experiment.py` 运行生成，全部数字来自当次运行的 "
      "`exports/selection_experiment/results.json`，不复算不得手改。")
    A("")
    A("## 设置")
    A("")
    A("- 同一逆变换原型：`spectralpaint.dsp.stft → 实数增益 → istft`（与编辑器引擎完全相同的"
      " rfft/OLA 路径），两种表示只是**选区的存储形式**不同。")
    A("- 创建窗 W1：n_fft=2048, hop=512, hann（频箱 "
      f"{SR / W1.n_fft:.3f} Hz，帧步 {W1.hop / SR * 1e3:.2f} ms）；"
      f"重开窗 W2：n_fft=4096, hop=1024, hann（频箱 {SR / W2.n_fft:.3f} Hz，帧步 "
      f"{W2.hop / SR * 1e3:.2f} ms）。")
    A("- 候选表示：")
    A("  - `curve`：连续选区存为曲线（f_lo(t)、f_hi(t) 折线，秒/Hz 物理单位），求值时再栅格化；")
    A("  - `mask_pixel`：带原始窗参数的系数掩码，按像素（频箱/帧序号）固化重开；")
    A("  - `mask_resample`：同一掩码按物理坐标线性重采样到新网格；")
    A("  - `mask_strict`：同一掩码，网格不符即拒绝重开。")
    A("- 显示网格（画布像素）在任何一条路径中都不参与存储或求值：坐标只有秒/Hz 或"
      "绑定显式窗参数的 STFT 单元。")
    A("")
    A("## A. 窄带切口（变窗后对相邻频带的影响）")
    A("")
    A(f"素材：{nb['signal']}。意图：{NOTCH.t0}–{NOTCH.t1} s / {NOTCH.f0}–{NOTCH.f1} Hz "
      f"施加 {NOTCH.gain_db} dB（σ_t={NOTCH.feather_t} s, σ_f={NOTCH.feather_f} Hz）。")
    A("")
    r = nb["reference_W1"]["bounds"]
    A(f"W1 参考（curve@W1）半增益边界：{r['f_lo']:.1f}–{r['f_hi']:.1f} Hz / "
      f"{r['t_lo']:.3f}–{r['t_hi']:.3f} s；1000 Hz 目标带衰减 "
      f"{nb['reference_W1']['att_1000_db']:.2f} dB，500 Hz 相邻带变化 "
      f"{nb['reference_W1']['adj_500_db']:.3f} dB。")
    A("")
    A("| 表示@W2 | 半增益频率边界 (Hz) | 边界误差 vs 意图 (Hz) | 漂移 vs W1 参考 Δf_lo/Δf_hi (Hz) "
      "| 1000 Hz 目标带 (dB) | 500 Hz 相邻带 (dB) |")
    A("|---|---|---|---|---|---|")
    for name in ("curve", "mask_pixel", "mask_resample", "mask_strict"):
        row = nb["W2"][name]
        if "error" in row:
            A(f"| {name} | 拒绝重开 | — | — | — | — |")
            continue
        m = row["measured"]
        A(f"| {name} | {m['f_lo']:.1f}–{m['f_hi']:.1f} | "
          f"{row['err_f_vs_intent']:.1f} | "
          f"{row['d_f_lo']:+.1f} / {row['d_f_hi']:+.1f} | "
          f"{row['att_1000_db']:.2f} | {row['adj_500_db']:.2f} |")
    A("")
    A("注：半增益边界相对硬边意图存在 ±1.18σ 的羽化偏移（设计如此，两种表示相同），"
      "因此“误差 vs 意图”含该固定偏移；表示之间的公平比较看“漂移 vs W1 参考”与"
      "相邻带/目标带两列。")
    A("")
    A("掩码边界见图 `exports/selection_experiment/figures/narrowband_mask_boundary_w2.png`，"
      "重建后 0.5–1.5 s 再分析谱见 `narrowband_spectra_w2.png`；可听对照 "
      "`narrowband_w2_*.wav`。")
    A("")
    A("## B. 瞬态（变窗后时间边界的漂移）")
    A("")
    A(f"素材：{tr['signal']}。意图：{TRANSIENT_EDIT.t0}–{TRANSIENT_EDIT.t1} s / "
      f"{TRANSIENT_EDIT.f0}–{TRANSIENT_EDIT.f1} Hz 施加 {TRANSIENT_EDIT.gain_db} dB，"
      "只应压住 click@0.725s。")
    A("")
    rt = tr["reference_W1"]["bounds"]
    A(f"W1 参考半增益时间边界：{rt['t_lo']:.3f}–{rt['t_hi']:.3f} s；"
      f"各 click 电平变化（dB）："
      + ", ".join(f"{v:.2f}" for v in tr["reference_W1"]["clicks_db"]) + "。")
    A("")
    A("| 表示@W2 | 半增益时间边界 (s) | 边界误差 vs 意图 (s) | 漂移 vs W1 参考 Δt_lo/Δt_hi (s) | click@0.30 | "
      "click@0.725 | click@1.15 | click@1.575 |")
    A("|---|---|---|---|---|---|---|---|")
    for name in ("curve", "mask_pixel", "mask_resample", "mask_strict"):
        row = tr["W2"][name]
        if "error" in row:
            A(f"| {name} | 拒绝重开 | — | — | — | — | — | — |")
            continue
        m = row["measured"]
        cl = row["clicks_db"]
        A(f"| {name} | {m['t_lo']:.3f}–{m['t_hi']:.3f} | "
          f"{row['err_t_vs_intent']:.3f} | "
          f"{row['d_t_lo']:+.3f} / {row['d_t_hi']:+.3f} | "
          + " | ".join(f"{v:.2f}" for v in cl) + " |")
    A("")
    A("波形与逐 click 电平见图 `exports/selection_experiment/figures/transient_w2.png`；"
      "可听对照 `transient_w2_*.wav`。")
    A("")
    A("## C. 撤销成本")
    A("")
    A(f"- 每条选区持久化体积：curve **{undo['curve_storage_bytes']} B** "
      f"（控制点折线）vs 系数掩码 **{undo['mask_storage_bytes']:,} B** "
      f"（{undo['mask_grid'][0]} 帧 × {undo['mask_grid'][1]} 箱 × float64）。")
    A(f"- 每次重建的求值耗时（W1 网格，60 次均值）：curve 重栅格化 "
      f"{undo['curve_eval_ms_per_rebuild']:.3f} ms vs 掩码直接取用 "
      f"{undo['mask_eval_ms_per_rebuild']:.3f} ms。")
    A("- 两者的撤销语义相同：从列表删除选区后由原始谱整体重建（测试套件逐位验证），"
      "不存在反向增益误差累积；差异只在持久体积与每次重建的求值耗时。")
    A("")
    A("## 结论：存储策略")
    A("")
    A("1. **以连续边界（曲线，秒/Hz 物理单位）为选区的唯一真实存储。** 变窗后在当前网格上"
      "重新求值，掩码边界漂移不超过新网格的量化步长（见 A/B 表 curve 行），目标带与相邻带"
      "行为与 W1 参考一致。")
    A("2. **系数掩码只能作为派生缓存，且必须绑定生成它的窗参数。** 跨窗按像素固化重开会把"
      "切口平移到别的频带/时段：本实验中 940–1060 Hz 切口漂到 "
      f"{nb['W2']['mask_pixel']['measured']['f_lo']:.0f}–"
      f"{nb['W2']['mask_pixel']['measured']['f_hi']:.0f} Hz，目标 1000 Hz 几乎未衰减而"
      "相邻 500 Hz 被误伤；瞬态编辑的 0.5–1.0 s 窗口漂到 "
      f"{tr['W2']['mask_pixel']['measured']['t_lo']:.2f}–"
      f"{tr['W2']['mask_pixel']['measured']['t_hi']:.2f} s，误伤 click@1.15/1.575。"
      "因此跨窗重开只有两种可接受策略：拒绝（strict）或重采样（resample，边界仍带着旧网格"
      "量化误差，且体积大三个数量级）。")
    A("3. **精度**：曲线以 float64 存边界，精度不受任何网格限制；掩码的边界在创建时即被"
      "量化到旧窗频箱（W1 下 ±10.8 Hz / ±5.8 ms），任何重开策略都无法找回。")
    A("4. **撤销成本**：曲线每条选区几十字节、重建时重栅格化约零点几毫秒；掩码每条选区"
      "约 1.4 MB、重建时零求值但换窗即失效。编辑会话的撤销栈应存曲线（或等价的物理单位"
      "笔画），掩码只在导出/渲染管线末端按需生成。")
    A("5. **显示网格不是音频真实坐标**：本实验全部测量在物理坐标（秒/Hz）或绑定显式窗参数"
      "的 STFT 单元上进行；画布像素只负责显示，任何“按显示像素固化选区”的做法等价于"
      " mask_pixel，变窗/缩放即错位。")
    A("")
    A("## 复算")
    A("")
    A("```bash")
    A(".venv/bin/python scripts/run_selection_experiment.py")
    A("```")
    A("")
    A("重新生成 `exports/selection_experiment/results.json`、全部 WAV/图以及本文件。")
    A("")
    return "\n".join(L)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)

    nb, x_nb, nb_fields = run_narrowband()
    tr, x_tr, tr_fields = run_transient()
    undo = run_undo_cost(x_nb.shape[0])
    make_figures(x_nb, x_tr, nb_fields, tr_fields)

    results = {"sr": SR, "W1": vars(W1), "W2": vars(W2),
               "narrowband": nb, "transient": tr, "undo_cost": undo}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    DOC.write_text(render_report(nb, tr, undo))
    print(json.dumps({"narrowband_W2": {k: v for k, v in nb["W2"].items()},
                      "transient_W2": {k: v for k, v in tr["W2"].items()},
                      "undo_cost": undo}, indent=2))
    print(f"\nwrote {OUT/'results.json'}\nwrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
