"""Local HTTP backend for the browser-based spectral editor.

Only Python's standard-library http.server is used, so the GUI runs in any
browser without a native toolkit (the container has no Tk/Qt libraries).  All
audio math stays in :mod:`spectralpaint`; this module is a thin JSON layer.

Endpoints
---------
GET  /                        static single-page UI
POST /api/session/load        load an uploaded audio file (or a built-in signal)
POST /api/session/demo        create one of dual_tone / sweep / transient / mixed
GET  /api/session             metadata, spectrogram image, gain field, coherence
POST /api/edit                add a stroke (physical units s / Hz)
POST /api/undo                exact undo (drop last stroke, rebuild from pristine)
POST /api/clear               remove all strokes
POST /api/rebuild             run inverse STFT, return metrics + edited spectrogram
GET  /api/audio?which=...     pristine or edited audio as WAV (for A/B preview)
POST /api/export              write edited audio to disk, then re-analyze the file
"""

from __future__ import annotations

import io
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

from .audioio import load_audio
from .dsp import STFTConfig
from .engine import EditSession
from .mask import Stroke
from .metrics import full_report
from . import signals as sigmod

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class SessionState:
    """Holds the live edit session behind a lock (server is threaded)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.session: EditSession | None = None
        self.edited: np.ndarray | None = None
        self.last_report: dict | None = None
        self.export_dir = Path.cwd() / "exports"
        self.export_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------- lifecycle
    def new(self, samples, sr, cfg, strategy, threshold):
        with self.lock:
            self.session = EditSession(samples, sr, cfg,
                                       strategy=strategy,
                                       coherence_threshold=threshold)
            self.edited = None
            self.last_report = None

    def require(self) -> EditSession:
        if self.session is None:
            raise RuntimeError("no audio loaded")
        return self.session


STATE = SessionState()


# --------------------------------------------------------------------- images
def _spectrogram_png(session, spec, gain_overlay, coherence) -> bytes:
    """Render a log-magnitude spectrogram (with optional gain/coherence overlay).

    Uses matplotlib's Agg backend (no display required).  The image is only a
    *view*; the underlying edit always operates on physical STFT cells.  Axes
    are labeled in physical units (seconds / Hz), not frame/bin indices.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, freqs = session.times, session.freqs
    extent = [times[0], times[-1] + (times[1] - times[0]),
              freqs[0], freqs[-1] + (freqs[1] - freqs[0])]
    mag = 20 * np.log10(np.abs(spec) + 1e-8)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
    im = ax.imshow(mag.T, origin="lower", aspect="auto", cmap="magma",
                   interpolation="nearest", extent=extent)
    fig.colorbar(im, ax=ax, label="magnitude (dB)")
    if gain_overlay is not None:
        # contour the gain field so the *actual* edited cells are visible
        db = 20 * np.log10(gain_overlay + 1e-12)
        levels = [lvl for lvl in (-24, -12, -6, -3, 3, 6, 12)
                  if db.min() < lvl < db.max()]
        if levels:
            cs = ax.contour(times, freqs, db.T, levels=levels, colors="cyan",
                            linewidths=0.7, alpha=0.9)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%d dB")
    if coherence is not None:
        ax.imshow(np.ma.masked_less(coherence.T, 0.5), origin="lower",
                  aspect="auto", cmap="cool", alpha=0.25, interpolation="nearest",
                  extent=extent)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("complex-spectrum magnitude + applied gain contours")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


# -------------------------------------------------------------------- helpers
def _cfg_from_body(body: dict) -> STFTConfig:
    return STFTConfig(
        n_fft=int(body.get("n_fft", 2048)),
        hop=int(body.get("hop", 512)),
        window=str(body.get("window", "hann")),
    )


def _session_info(include_images: bool) -> dict:
    s = STATE.require()
    info = {
        "sr": s.sr,
        "n_samples": int(s.samples.shape[0]),
        "duration": s.samples.shape[0] / s.sr,
        "n_channels": s.n_channels,
        "n_frames": s.n_frames,
        "n_bins": s.n_bins,
        "n_fft": s.cfg.n_fft,
        "hop": s.cfg.hop,
        "window": s.cfg.window,
        "strategy": s.strategy,
        "coherence_threshold": s.coherence_threshold,
        "times": s.times.tolist(),
        "freqs": s.freqs.tolist(),
        "strokes": [vars(st) for st in s.strokes],
        "gain_db": (20 * np.log10(s.gain_field() + 1e-12)).tolist(),
        "coherence": s.coherence().tolist(),
    }
    if include_images:
        info["spectrogram_png"] = "data:image/png;base64," + __import__("base64").b64encode(
            _spectrogram_png(s, s._base_specs[0], s.gain_field(),
                             s.coherence() if s.strategy == "coherent_joint" else None)
        ).decode()
        if STATE.edited is not None:
            from .dsp import stft
            info["edited_spectrogram_png"] = "data:image/png;base64," + __import__("base64").b64encode(
                _spectrogram_png(s, stft(STATE.edited[:, 0], s.cfg), None, None)
            ).decode()
    if STATE.last_report is not None:
        info["report"] = STATE.last_report
    return info


# --------------------------------------------------------------------- routes
def handle_load(body: dict) -> dict:
    cfg = _cfg_from_body(body)
    strategy = body.get("strategy", "phase_locked")
    threshold = float(body.get("coherence_threshold", 0.6))
    if "_audio" in body:
        samples, sr = body["_audio"]
    elif "demo" in body:
        fn = getattr(sigmod, body["demo"])
        samples, sr = fn()
    else:
        raise RuntimeError("use /api/session/demo or multipart upload")
    STATE.new(samples, sr, cfg, strategy, threshold)
    return _session_info(True)


def handle_edit(body: dict) -> dict:
    s = STATE.require()
    stroke = Stroke(
        t0=float(body["t0"]), t1=float(body["t1"]),
        f0=float(body["f0"]), f1=float(body["f1"]),
        gain_db=float(body["gain_db"]),
        feather_t=float(body.get("feather_t", 0.03)),
        feather_f=float(body.get("feather_f", 60.0)),
    )
    with STATE.lock:
        s.add_stroke(stroke)
        STATE.edited = None
        STATE.last_report = None
    return _session_info(True)


def handle_undo() -> dict:
    s = STATE.require()
    with STATE.lock:
        removed = s.undo()
        STATE.edited = None
        STATE.last_report = None
    return {"removed": vars(removed) if removed else None,
            **_session_info(True)}


def handle_clear() -> dict:
    s = STATE.require()
    with STATE.lock:
        s.clear_edits()
        STATE.edited = None
        STATE.last_report = None
    return _session_info(True)


def handle_rebuild() -> dict:
    s = STATE.require()
    with STATE.lock:
        edited = s.rebuild()
        STATE.edited = edited
        STATE.last_report = full_report(s, edited)
    return _session_info(True)


def handle_export(body: dict) -> dict:
    s = STATE.require()
    if STATE.edited is None:
        handle_rebuild()
    name = body.get("name", "edited.wav")
    out = STATE.export_dir / Path(name).name  # name only, no path traversal
    sf.write(str(out), STATE.edited, s.sr, subtype="FLOAT")
    # Re-analyze the exported file to prove what was written on disk matches
    # the in-memory reconstruction (export -> reload -> measure).
    reloaded, rsr = load_audio(str(out))
    reanalyzed = {
        "path": str(out),
        "sample_rate": rsr,
        "shape": list(reloaded.shape),
        "matches_in_memory": bool(np.allclose(reloaded, STATE.edited, atol=1e-6)),
        "peak": float(np.max(np.abs(reloaded))),
        "rms": float(np.sqrt(np.mean(reloaded ** 2))),
    }
    from .dsp import stft
    re_spec = stft(reloaded[:, 0], s.cfg)
    reanalyzed["spectrogram_png"] = "data:image/png;base64," + __import__("base64").b64encode(
        _spectrogram_png(s, re_spec, None, None)).decode()
    return {"export": reanalyzed, **_session_info(False)}


def handle_strategy(body: dict) -> dict:
    s = STATE.require()
    with STATE.lock:
        s.strategy = body.get("strategy", s.strategy)
        s.coherence_threshold = float(body.get("coherence_threshold",
                                                s.coherence_threshold))
        STATE.edited = None
        STATE.last_report = None
    return _session_info(True)


class Handler(BaseHTTPRequestHandler):
    server_version = "SpectralPaint/1.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, default=_json_default).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/" or path == "/index.html":
                self._serve_static("index.html", "text/html")
            elif path.startswith("/static/"):
                self._serve_static(Path(path).name, _mime(Path(path).name))
            elif path == "/api/session":
                qs = urllib.parse.parse_qs(parsed.query)
                include = qs.get("img", ["1"])[0] != "0"
                self._send_json(_session_info(include_images=include))
            elif path == "/api/audio":
                qs = urllib.parse.parse_qs(parsed.query)
                self._serve_audio(qs.get("which", ["edited"])[0])
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001 - surface as JSON
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            ctype = self.headers.get("Content-Type", "")
            if "application/json" in ctype or raw.startswith(b"{"):
                body = json.loads(raw or b"{}")
            else:
                body = _parse_multipart(raw, ctype)
            route = {
                "/api/session/load": lambda: handle_load(body),
                "/api/session/demo": lambda: handle_load(body),
                "/api/edit": lambda: handle_edit(body),
                "/api/undo": lambda: handle_undo(),
                "/api/clear": lambda: handle_clear(),
                "/api/rebuild": lambda: handle_rebuild(),
                "/api/export": lambda: handle_export(body),
                "/api/strategy": lambda: handle_strategy(body),
            }.get(parsed.path)
            if route is None:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send_json(route())
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    # ------------------------------------------------------------- static/audio
    def _serve_static(self, name: str, ctype: str):
        p = (WEB_DIR / name).resolve()
        if WEB_DIR not in p.parents and p != WEB_DIR / name:
            self._send_json({"error": "forbidden"}, 403)
            return
        if not p.exists():
            self._send_json({"error": "missing static file"}, 404)
            return
        self._send_bytes(p.read_bytes(), ctype)

    def _serve_audio(self, which: str):
        s = STATE.require()
        if which == "pristine":
            data = _wav_bytes(s.pristine(), s.sr)
        else:
            if STATE.edited is None:
                handle_rebuild()
            data = _wav_bytes(STATE.edited, s.sr)
        self._send_bytes(data, "audio/wav")


def _mime(name: str) -> str:
    return {".html": "text/html", ".js": "application/javascript",
            ".css": "text/css"}.get(Path(name).suffix, "application/octet-stream")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _parse_multipart(raw: bytes, ctype: str) -> dict:
    """Minimal multipart parser: extract a WAV file plus JSON form fields."""
    import email
    msg = email.message_from_bytes(
        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
    body: dict = {}
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        name = part.get_param("name", header="content-disposition")
        if name == "audio":
            data = part.get_payload(decode=True)
            audio, sr = sf.read(io.BytesIO(data), always_2d=True,
                                dtype="float64")
            body["_audio"] = (audio, sr)
        elif name:
            body[name] = part.get_payload()
    return body


def serve(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd
