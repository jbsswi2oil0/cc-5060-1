/* Spectral smearing editor front-end.
 *
 * The canvas never re-interprets pixels as frequency bins.  It draws the
 * server-provided gain/magnitude *grids* through an explicit view transform
 * (t0,t1,f0,f1 in seconds/Hz).  A drag is converted back to physical units and
 * sent to the backend, which rasterizes onto the actual STFT cells -- zooming
 * only changes the view, never the sound.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("spec"), ctx = canvas.getContext("2d");
const tooltip = $("tooltip");

let S = null;                 // last session info from server
let view = { t0: 0, t1: 1, f0: 0, f1: 1000 };
let drag = null;               // {x0,y0,x1,y1} in CSS pixels
let spectroImg = null;         // base spectrogram image
let editedImg = null;

async function api(path, body) {
  const opt = body === undefined
    ? { method: "GET" }
    : { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

// ------------------------------------------------------------- view mapping
function cssSize() { return { w: canvas.clientWidth, h: canvas.clientHeight }; }
function xToT(x) { const { w } = cssSize(); return view.t0 + (x / w) * (view.t1 - view.t0); }
function yToF(y) { const { h } = cssSize(); return view.f1 - (y / h) * (view.f1 - view.f0); }
function tToX(t) { const { w } = cssSize(); return (t - view.t0) / (view.t1 - view.t0) * w; }
function fToY(f) { const { h } = cssSize(); return (view.f1 - f) / (view.f1 - view.f0) * h; }

function resetView() {
  if (!S) return;
  view = { t0: 0, t1: S.duration, f0: 0, f1: S.freqs[S.freqs.length - 1] };
}

// --------------------------------------------------------------- rendering
function loadImage(dataUrl) {
  return new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.src = dataUrl; });
}

async function render() {
  if (!S) return;
  const { w, h } = cssSize();
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, w, h);
  if (spectroImg && S.spectrogram_png) {
    // base spectrogram covers full data extent; draw the visible sub-rect
    drawDataImage(spectroImg);
  }
  drawGainOverlay();
  drawGrid();
  drawStrokes();
  if (drag) drawDragRect();
}

function drawDataImage(img) {
  const { w, h } = cssSize();
  // image spans the full t/f data range; map the current view window onto it
  const u0 = view.t0 / S.duration, u1 = view.t1 / S.duration;
  const fMax = S.freqs[S.freqs.length - 1];
  const v1 = 1 - view.f0 / fMax, v0 = 1 - view.f1 / fMax; // image y is top-down
  ctx.globalAlpha = 0.85;
  ctx.drawImage(img, u0 * img.width, v0 * img.height,
    (u1 - u0) * img.width, (v1 - v0) * img.height, 0, 0, w, h);
  ctx.globalAlpha = 1;
}

function drawGainOverlay() {
  // Heat-map the actual per-cell gain field (the real edit), not a fake tint.
  const g = S.gain_db;
  const nT = g.length, nF = S.freqs.length;
  const tMin = view.t0, tMax = view.t1, fMin = view.f0, fMax = view.f1;
  const i0 = Math.max(0, binIndex(S.times, tMin) - 1);
  const i1 = Math.min(nT - 1, binIndex(S.times, tMax) + 1);
  const j0 = Math.max(0, binIndex(S.freqs, fMin) - 1);
  const j1 = Math.min(nF - 1, binIndex(S.freqs, fMax) + 1);
  for (let i = i0; i <= i1; i++) {
    for (let j = j0; j <= j1; j++) {
      const db = g[i][j];
      if (Math.abs(db) < 0.05) continue;
      const x = tToX(S.times[i]), y = fToY(S.freqs[j]);
      const cw = wCell(i), ch = hCell(j);
      ctx.fillStyle = db < 0
        ? `rgba(255,90,90,${Math.min(0.55, -db / 40)})`
        : `rgba(90,220,140,${Math.min(0.55, db / 24)})`;
      ctx.fillRect(x - cw / 2, y - ch / 2, cw + 1, ch + 1);
    }
  }
}
function binIndex(axis, v) {
  let lo = 0, hi = axis.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (axis[m] < v) lo = m + 1; else hi = m; }
  return lo;
}
function wCell(i) {
  const { w } = cssSize();
  const dt = S.n_frames > 1 ? S.times[1] - S.times[0] : S.duration;
  return Math.max(1, dt / (view.t1 - view.t0) * w);
}
function hCell(j) {
  const { h } = cssSize();
  const df = S.freqs[1] - S.freqs[0];
  return Math.max(1, df / (view.f1 - view.f0) * h);
}

function drawGrid() {
  const { w, h } = cssSize();
  ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.fillStyle = "#9aa3b2";
  ctx.font = "10px monospace"; ctx.lineWidth = 1;
  // time ticks (choose ~8 nice steps)
  const ticksT = niceTicks(view.t0, view.t1, 8);
  ticksT.forEach((t) => { const x = tToX(t); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillText(t.toFixed(2) + "s", x + 3, h - 5); });
  const ticksF = niceTicks(view.f0, view.f1, 6);
  ticksF.forEach((f) => { const y = fToY(f); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.fillText(f < 1000 ? f.toFixed(0) + "Hz" : (f / 1000).toFixed(2) + "k", 4, y - 3); });
}

function drawStrokes() {
  const hard = S.strokes;
  ctx.lineWidth = 1.5;
  hard.forEach((st, k) => {
    const x = tToX(st.t0), y = fToY(st.f1), wdt = tToX(st.t1) - tToX(st.t0), hgt = fToY(st.f0) - fToY(st.f1);
    ctx.strokeStyle = "#4cc2ff"; ctx.setLineDash([5, 4]);
    ctx.strokeRect(x, y, wdt, hgt); ctx.setLineDash([]);
    // feather extent (3 sigma) shown as a faint solid outline
    const fx = tToX(Math.max(0, st.t0 - 3 * st.feather_t));
    const fx1 = tToX(Math.min(S.duration, st.t1 + 3 * st.feather_t));
    const fy = fToY(Math.min(S.freqs.at(-1), st.f1 + 3 * st.feather_f));
    const fy0 = fToY(Math.max(0, st.f0 - 3 * st.feather_f));
    ctx.strokeStyle = "rgba(76,194,255,0.35)";
    ctx.strokeRect(fx, fy, fx1 - fx, fy0 - fy);
    ctx.fillStyle = "#4cc2ff"; ctx.fillText("#" + k, x + 3, y + 12);
  });
}

function drawDragRect() {
  const { x0, y0, x1, y1 } = drag;
  ctx.strokeStyle = "#ffd166"; ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
  ctx.setLineDash([]);
}

function niceTicks(lo, hi, n) {
  const span = hi - lo, raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
  return out;
}

// ------------------------------------------------------------- session sync
async function refresh(info, keepImg) {
  S = info;
  $("hdrInfo").textContent =
    `${S.n_channels}ch · ${S.sr}Hz · ${S.duration.toFixed(2)}s · nfft=${S.n_fft} hop=${S.hop} ${S.window} · ${S.n_frames}帧×${S.n_bins}箱`;
  if (!keepImg || !spectroImg) {
    if (S.spectrogram_png) spectroImg = await loadImage(S.spectrogram_png);
    resetView();
  }
  if (S.edited_spectrogram_png) editedImg = await loadImage(S.edited_spectrogram_png);
  renderStrokeTable();
  renderMetrics();
  render();
}

function renderStrokeTable() {
  const tb = $("strokeTable"); tb.innerHTML = "";
  S.strokes.forEach((st, k) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${k}</td><td>${st.t0.toFixed(3)}–${st.t1.toFixed(3)}</td>
      <td>${st.f0.toFixed(0)}–${st.f1.toFixed(0)}</td><td class="num">${st.gain_db}</td>`;
    tb.appendChild(tr);
  });
}

function renderMetrics() {
  const el = $("metrics");
  if (!S.report) { el.className = "muted"; el.textContent = "尚未重建。"; return; }
  const r = S.report;
  const rows = [];
  rows.push(["编辑笔数", r.n_strokes, ""]);
  rows.push(["策略", r.strategy, ""]);
  if (r.effective_bounds)
    rows.push(["作用范围", `${r.effective_bounds.t0.toFixed(2)}–${r.effective_bounds.t1.toFixed(2)}s / ${r.effective_bounds.f0.toFixed(0)}–${r.effective_bounds.f1.toFixed(0)}Hz`, ""]);
  rows.push(["时域 ΔRMS", r.time_domain.relative_rms_db.toFixed(2) + " dB", ""]);
  const leak = r.leakage.max_leak_energy_ratio;
  rows.push(["未触及单元泄漏", leak.toExponential(2), leak < 1e-3 ? "ok" : "warn"]);
  if (r.pan) {
    rows.push(["声像偏移(精确谱)", r.pan.pan_shift_rms_db.toExponential(2) + " dB",
      Math.abs(r.pan.pan_shift_rms_db) < 1e-6 ? "ok" : "warn"]);
    rows.push(["声像偏移(再分析)", r.pan_reanalyzed.pan_shift_rms_db.toFixed(3) + " dB", ""]);
  }
  if (r.phase) {
    rows.push(["相位差变化(精确谱)", r.phase.phase_diff_change_rms_rad.toExponential(2) + " rad",
      r.phase.phase_diff_change_rms_rad < 1e-9 ? "ok" : "warn"]);
    rows.push(["相位差变化(再分析)", r.phase_reanalyzed.phase_diff_change_rms_rad.toExponential(2) + " rad", ""]);
  }
  r.boundary.per_channel.forEach((b, c) =>
    rows.push([`边界残留 ch${c}`, b.ring_energy_change_db.toFixed(2) + " dB", ""]));
  el.className = "";
  el.innerHTML = `<table><tbody>${rows.map(([k, v, cls]) =>
    `<tr><td class="muted">${k}</td><td class="num ${cls}">${v}</td></tr>`).join("")}</tbody></table>`;
}

// ------------------------------------------------------------------ events
canvas.addEventListener("mousedown", (e) => {
  const rct = canvas.getBoundingClientRect();
  drag = { x0: e.clientX - rct.left, y0: e.clientY - rct.top, x1: e.clientX - rct.left, y1: e.clientY - rct.top };
});
canvas.addEventListener("mousemove", async (e) => {
  const rct = canvas.getBoundingClientRect();
  const x = e.clientX - rct.left, y = e.clientY - rct.top;
  tooltip.style.display = "block";
  tooltip.style.left = e.clientX + 12 + "px"; tooltip.style.top = e.clientY + 12 + "px";
  tooltip.textContent = `t=${xToT(x).toFixed(4)}s  f=${yToF(y).toFixed(1)}Hz`;
  if (drag) { drag.x1 = x; drag.y1 = y; render(); }
});
canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
canvas.addEventListener("mouseup", async () => {
  if (!drag || !S) return;
  const t0 = xToT(Math.min(drag.x0, drag.x1)), t1 = xToT(Math.max(drag.x0, drag.x1));
  const f0 = yToF(Math.max(drag.y0, drag.y1)), f1 = yToF(Math.min(drag.y0, drag.y1));
  drag = null;
  if (t1 - t0 < 0.005 || f1 - f0 < 5) { render(); return; } // ignore clicks
  $("selReadout").value = `${t0.toFixed(3)}–${t1.toFixed(3)}s, ${f0.toFixed(0)}–${f1.toFixed(0)}Hz`;
  try {
    const info = await api("/api/edit", {
      t0, t1, f0, f1, gain_db: parseFloat($("gainDb").value),
      feather_t: parseFloat($("featherT").value),
      feather_f: parseFloat($("featherF").value),
    });
    await refresh(info, true);
    setStatus("已添加选区（尚未重建波形）。");
  } catch (err) { setStatus(err.message, true); }
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rct = canvas.getBoundingClientRect();
  const x = e.clientX - rct.left, y = e.clientY - rct.top;
  const fcT = xToT(x), fcF = yToF(y);
  const z = e.deltaY > 0 ? 1.18 : 1 / 1.18;
  if (e.shiftKey) {
    zoomAxis(view, "f", fcF, z);
  } else {
    zoomAxisT(view, fcT, z);
    zoomAxis(view, "f", fcF, z);
  }
  render();
}, { passive: false });
canvas.addEventListener("dblclick", () => { resetView(); render(); });

function zoomAxisT(v, center, z) {
  v.t0 = center - (center - v.t0) * z; v.t1 = center + (v.t1 - center) * z;
  if (v.t0 < 0) v.t0 = 0;
}
function zoomAxis(v, axis, center, z) {
  v.f0 = center - (center - v.f0) * z; v.f1 = center + (v.f1 - center) * z;
  if (v.f0 < 0) v.f0 = 0;
}

// ---------------------------------------------------------------- buttons
async function loadDemo(demo) {
  try {
    const info = await api("/api/session/demo", {
      demo, n_fft: +$("nFft").value, hop: +$("hop").value, window: $("window").value,
      strategy: $("strategy").value, coherence_threshold: +$("cohThr").value,
    });
    await refresh(info); setStatus(`已载入内置样例：${demo}`);
  } catch (err) { setStatus(err.message, true); }
}
$("btnDual").onclick = () => loadDemo("dual_tone");
$("btnSweep").onclick = () => loadDemo("sweep");
$("btnTransient").onclick = () => loadDemo("transient_mix");
$("btnMixed").onclick = () => loadDemo("mixed_sample");

$("fileInput").onchange = async (e) => {
  const fd = new FormData();
  fd.append("audio", e.target.files[0]);
  fd.append("n_fft", $("nFft").value); fd.append("hop", $("hop").value);
  fd.append("window", $("window").value); fd.append("strategy", $("strategy").value);
  fd.append("coherence_threshold", $("cohThr").value);
  try {
    const r = await fetch("/api/session/load", { method: "POST", body: fd });
    const j = await r.json(); if (j.error) throw new Error(j.error);
    await refresh(j); setStatus("已载入文件：" + e.target.files[0].name);
  } catch (err) { setStatus(err.message, true); }
};

$("applyCfg").onclick = async () => {
  // Window parameters change the STFT grid.  Selections are stored as
  // continuous s/Hz boundaries (see docs/selection_representation.md), so the
  // backend re-evaluates them on the new grid instead of dropping them.
  // Kernel errors (e.g. hop > n_fft) come back as JSON and are shown here.
  try {
    const j = await api("/api/reconfigure", {
      n_fft: +$("nFft").value, hop: +$("hop").value, window: $("window").value,
      strategy: $("strategy").value, coherence_threshold: +$("cohThr").value,
    });
    await refresh(j);
    const r = j.reconfigured;
    setStatus(`分析窗已切换为 nfft=${r.n_fft} hop=${r.hop} ${r.window}；`
      + `${r.kept_strokes} 个选区按连续边界在新网格重算（未丢弃）。`);
  } catch (err) { setStatus("重建会话失败（内核拒绝参数）：" + err.message, true); }
};

$("undo").onclick = async () => {
  try { const j = await api("/api/undo", {}); await refresh(j, true); setStatus("已撤销最后一笔（从原始谱重建，非反向增益）。"); }
  catch (err) { setStatus(err.message, true); }
};
$("clear").onclick = async () => {
  try { const j = await api("/api/clear", {}); await refresh(j, true); setStatus("已清空全部编辑。"); }
  catch (err) { setStatus(err.message, true); }
};
$("rebuild").onclick = async () => {
  try { const j = await api("/api/rebuild", {}); await refresh(j); setStatus("已通过逆 STFT + 重叠相加重建波形。"); }
  catch (err) { setStatus(err.message, true); }
};
$("export").onclick = async () => {
  const name = "edited_" + Date.now() + ".wav";
  try { const j = await api("/api/export", { name });
    await refresh(j, true);
    setStatus(`已导出并再分析：${j.export.path}（与内存重建一致=${j.export.matches_in_memory}）`);
  } catch (err) { setStatus(err.message, true); }
};

// A/B preview uses the same backend reconstruction; headless users can save via the player
function play(which) {
  const p = $("player");
  p.src = "/api/audio?which=" + which + "&t=" + Date.now();
  p.play().catch(() => setStatus("当前环境无音频输出设备，可用播放器下载按钮试听。", true));
  document.querySelectorAll(".ab button").forEach((b) => b.classList.remove("active"));
  (which === "pristine" ? $("playPristine") : $("playEdited")).classList.add("active");
}
$("playPristine").onclick = () => play("pristine");
$("playEdited").onclick = () => play("edited");
$("stopPlay").onclick = () => $("player").pause();

function setStatus(msg, warn) {
  const el = $("status"); el.textContent = msg; el.className = warn ? "warn" : "ok";
}
window.addEventListener("resize", render);
