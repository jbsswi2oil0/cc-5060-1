"""Launcher / CLI for the spectral smearing editor.

Usage
-----
Browser UI:
    python -m scripts.run_app serve --port 8765

Headless batch demonstration (also used as an end-to-end check):
    python -m scripts.run_app demo --signal mixed_sample --out exports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectralpaint.audioio import load_audio, write_float_wav
from spectralpaint.dsp import STFTConfig, istft, roundtrip_error, stft
from spectralpaint.engine import EditSession
from spectralpaint.mask import Stroke
from spectralpaint.metrics import full_report
from spectralpaint import signals as sigmod


def _signal(name: str):
    return getattr(sigmod, name)()


def run_demo(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = STFTConfig(n_fft=args.n_fft, hop=args.hop, window=args.window)
    samples, sr = _signal(args.signal)
    write_float_wav(out_dir / f"{args.signal}_pristine.wav", samples, sr)

    # round-trip must be exact before any edit is applied
    rt = roundtrip_error(samples[:, 0], cfg)

    session = EditSession(samples, sr, cfg, strategy=args.strategy,
                          coherence_threshold=args.coherence_threshold)
    for stroke in _demo_strokes(args.signal, sr):
        session.add_stroke(stroke)

    edited = session.rebuild()
    report = full_report(session, edited)
    report["roundtrip_no_edit"] = rt
    out_path = out_dir / f"{args.signal}_edited_{args.strategy}.wav"
    write_float_wav(out_path, edited, sr)

    # export -> reload -> re-analyze (proves the file on disk is the real edit)
    reloaded, rsr = load_audio(str(out_path))
    report["export"] = {
        "path": str(out_path), "sample_rate": rsr,
        "shape": list(reloaded.shape),
        "matches_in_memory": bool(np.allclose(reloaded, edited, atol=1e-6)),
    }
    report["roundtrip_no_edit"]["max_abs"] = rt["max_abs"]

    # exact undo check inside the batch flow
    n = len(session.strokes)
    for _ in range(n):
        session.undo()
    baseline = session.rebuild()
    no_edit_target = np.stack([
        istft(stft(samples[:, c], cfg), cfg, samples.shape[0])
        for c in range(samples.shape[1])], axis=1)
    report["exact_undo_bitexact"] = bool(np.array_equal(baseline, no_edit_target))
    # restore strokes for any downstream consumer
    for stroke in _demo_strokes(args.signal, sr):
        session.add_stroke(stroke)

    print(json.dumps(report, indent=2, default=str))
    return report


def _demo_strokes(name: str, sr: int) -> list[Stroke]:
    if name == "dual_tone":
        # attenuate only the 440 Hz tone; 1760 Hz must remain untouched
        return [Stroke(0.2, 1.8, 380, 500, -24.0, 0.02, 15.0)]
    if name == "sweep":
        return [Stroke(0.9, 1.2, 1200, 1800, -20.0, 0.01, 20.0)]
    if name == "transient_mix":
        return [Stroke(0.25, 0.45, 200, 4000, -15.0, 0.02, 40.0)]
    return [
        Stroke(0.4, 1.4, 250, 360, -22.0, 0.03, 30.0),     # low tone band
        Stroke(0.8, 2.2, 2300, 2700, 8.0, 0.03, 40.0),     # high tone boost
    ]


def serve(args) -> None:
    from spectralpaint.server import serve as make_server
    httpd = make_server(args.host, args.port)
    print(f"Spectral Paint UI: http://{args.host}:{args.port}")
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Spectral smearing audio editor")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="headless end-to-end demonstration")
    d.add_argument("--signal", default="mixed_sample",
                   choices=["dual_tone", "sweep", "transient_mix", "mixed_sample"])
    d.add_argument("--out", default="exports")
    d.add_argument("--n-fft", type=int, default=2048)
    d.add_argument("--hop", type=int, default=512)
    d.add_argument("--window", default="hann")
    d.add_argument("--strategy", default="phase_locked",
                   choices=["phase_locked", "coherent_joint"])
    d.add_argument("--coherence-threshold", type=float, default=0.6)
    d.set_defaults(func=run_demo)

    s = sub.add_parser("serve", help="launch browser UI backend")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=serve)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
