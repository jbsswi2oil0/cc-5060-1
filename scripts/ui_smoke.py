"""Browser smoke test for the spectral smearing editor UI.

Launches the real backend (spectralpaint.server) and drives the actual web UI
in headless Chromium via Playwright:

  1. kernel failure feedback: rebuild with no audio loaded -> error in status;
  2. load the dual-tone demo, drag a box on the canvas (box-select a frequency
     region) -> stroke registered in physical units;
  3. rebuild -> metrics table populated from the kernel report;
  4. A/B audio endpoints serve real WAV for listening comparison;
  5. reconfigure n_fft 2048 -> 4096 via the UI -> the selection survives
     (continuous-boundary storage re-evaluated on the new grid);
  6. screenshots saved to exports/figures/ui_*.png.

Run:  .venv/bin/python scripts/ui_smoke.py
Requires the Playwright chromium browser (see README "运行" section).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectralpaint.server import serve

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "exports" / "figures"


def main() -> int:
    from playwright.sync_api import sync_playwright

    httpd = serve("127.0.0.1", 0)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    FIG.mkdir(parents=True, exist_ok=True)
    checks = {}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(base, wait_until="networkidle")

            # 1) kernel failure must surface in the UI status line
            page.click("#rebuild")
            page.wait_for_function(
                "document.getElementById('status').textContent.length > 0")
            status = page.text_content("#status")
            checks["failure_feedback"] = "no audio loaded" in status

            # 2) load dual-tone demo
            page.click("#btnDual")
            page.wait_for_function(
                "document.getElementById('hdrInfo').textContent.includes('nfft=2048')")
            checks["demo_loaded"] = True

            # 3) box-select a frequency region on the canvas (drag)
            box = page.locator("#spec").bounding_box()
            x0, y0 = box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.75
            x1, y1 = box["x"] + box["width"] * 0.75, box["y"] + box["height"] * 0.85
            page.mouse.move(x0, y0)
            page.mouse.down()
            page.mouse.move(x1, y1, steps=12)
            page.mouse.up()
            page.wait_for_function(
                "document.querySelectorAll('#strokeTable tr').length === 1")
            readout = page.input_value("#selReadout")
            checks["box_select_physical_units"] = ("s," in readout
                                                   and "Hz" in readout)

            # 4) rebuild -> metrics table from the kernel report
            page.click("#rebuild")
            page.wait_for_function(
                "document.getElementById('metrics').textContent.includes('泄漏')")
            checks["metrics_rendered"] = True
            page.screenshot(path=str(FIG / "ui_smoke_edited.png"))

            # 5) A/B audio endpoints serve WAV (listening comparison)
            for which in ("pristine", "edited"):
                resp = page.request.get(f"{base}/api/audio?which={which}")
                checks[f"audio_{which}"] = (resp.status == 200
                                            and len(resp.body()) > 10000)

            # 6) reconfigure window via the UI: selection must survive
            page.select_option("#nFft", "4096")
            page.select_option("#hop", "1024")
            page.click("#applyCfg")
            page.wait_for_function(
                "document.getElementById('hdrInfo').textContent.includes('nfft=4096')")
            n_strokes = page.evaluate(
                "document.querySelectorAll('#strokeTable tr').length")
            status = page.text_content("#status")
            checks["reconfigure_keeps_selection"] = (
                n_strokes == 1 and "连续边界" in status)
            page.screenshot(path=str(FIG / "ui_smoke_reconfigured.png"))

            browser.close()
    finally:
        httpd.shutdown()

    checks["all_passed"] = all(checks.values())
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"screenshots: {FIG}/ui_smoke_edited.png, "
          f"{FIG}/ui_smoke_reconfigured.png")
    return 0 if checks["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
