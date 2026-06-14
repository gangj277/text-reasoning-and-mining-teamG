#!/usr/bin/env python3
"""
Playwright-based PDF exporter for slide decks.

Navigates to each slide-N.html standalone (not through the viewer), waits
for React + Babel to mount, screenshots the 1920×1080 slide region, and
combines every page into a single PDF.

Usage:
    python3 export_pdf.py                    # Export all slides (auto-starts server)
    python3 export_pdf.py --output deck.pdf  # Custom output filename
    python3 export_pdf.py --port 8731        # Custom port (default: auto-find)
    python3 export_pdf.py --no-server        # Don't auto-start server (expects one running)

Requirements:
    pip install playwright Pillow
    playwright install chromium
"""

import argparse
import glob
import os
import re
import socket
import subprocess
import sys
import time
from io import BytesIO

from PIL import Image
from playwright.sync_api import sync_playwright

# Wait until both fonts are ready AND the slide template has signalled
# that React has finished its first render. slide-template.html sets
# document.body.dataset.slideReady = '1' after ReactDOM.createRoot().render().
WAIT_READY_JS = """
() => Promise.all([
    document.fonts.ready,
    new Promise((resolve) => {
        if (document.body.dataset.slideReady === '1') return resolve(true);
        const obs = new MutationObserver(() => {
            if (document.body.dataset.slideReady === '1') {
                obs.disconnect();
                resolve(true);
            }
        });
        obs.observe(document.body, { attributes: true, attributeFilter: ['data-slide-ready'] });
        // Hard timeout fallback — some slides may not set the flag
        // (e.g., older decks). After 8s, assume ready.
        setTimeout(() => { obs.disconnect(); resolve(true); }, 8000);
    })
]).then(() => true)
"""

# In export mode we want the 1920×1080 .slide element filling the viewport
# exactly, with no viewer chrome, no transform scaling, and a transparent
# page background so the screenshot is pixel-perfect.
EXPORT_SETUP_JS = """
() => {
    // Neutralize the scale-to-viewport transform the template applies.
    const stage = document.getElementById('stage');
    if (stage) {
        stage.style.position = 'fixed';
        stage.style.top = '0';
        stage.style.left = '0';
        stage.style.transform = 'none';
    }
    const slide = document.getElementById('slide-root') || document.querySelector('.slide');
    if (slide) {
        slide.style.borderRadius = '0';
        slide.style.boxShadow = 'none';
    }
    document.documentElement.style.background = 'transparent';
    document.body.style.background = 'transparent';
    document.body.style.overflow = 'hidden';
}
"""


def list_slides(deck_dir):
    """Return sorted list of slide-N.html filenames."""
    files = glob.glob(os.path.join(deck_dir, "slide-*.html"))

    def slide_num(p):
        m = re.search(r"slide-(\d+)\.html", os.path.basename(p))
        return int(m.group(1)) if m else 0

    files.sort(key=slide_num)
    return [os.path.basename(f) for f in files]


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def find_available_port(start=8731):
    port = start
    while port < start + 100:
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"No available ports in range {start}-{start + 100}")


def start_server(deck_dir, port):
    proc = subprocess.Popen(
        [sys.executable, '-m', 'http.server', str(port)],
        cwd=deck_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if is_port_in_use(port):
            return proc
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError(f"Failed to start server on port {port}")


def export_slides_to_pdf(deck_dir, output_path, port=None, auto_start=True):
    slides = list_slides(deck_dir)
    if not slides:
        print("No slide files found!")
        sys.exit(1)

    print(f"Found {len(slides)} slide files")

    # Server setup
    server_proc = None
    if port and is_port_in_use(port):
        print(f"Using existing server on port {port}")
    elif auto_start:
        port = port or find_available_port()
        print(f"Starting HTTP server on port {port}...")
        server_proc = start_server(deck_dir, port)
    else:
        port = port or 8731
        if not is_port_in_use(port):
            print(f"No server running on port {port}.")
            sys.exit(1)

    base_url = f"http://localhost:{port}"
    screenshots = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=2,
            )

            for i, slide_file in enumerate(slides, start=1):
                print(f"  [{i:2d}/{len(slides)}] {slide_file}")

                url = f"{base_url}/{slide_file}"
                page.goto(url, wait_until="networkidle")

                # Wait for React mount + fonts.
                page.evaluate(WAIT_READY_JS)
                # Apply export-mode layout tweaks.
                page.evaluate(EXPORT_SETUP_JS)
                # One extra beat to let layout settle after style tweaks.
                page.wait_for_timeout(120)

                # Screenshot the full 1920×1080 viewport. We set export-mode
                # styles so the slide fills the viewport exactly.
                png = page.screenshot(
                    type="png",
                    clip={"x": 0, "y": 0, "width": 1920, "height": 1080},
                )

                img = Image.open(BytesIO(png))
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                screenshots.append(img)

            browser.close()

        if screenshots:
            screenshots[0].save(
                output_path,
                "PDF",
                resolution=300.0,
                save_all=True,
                append_images=screenshots[1:] or [],
            )
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n  PDF: {output_path}")
            print(f"  Size: {size_mb:.1f} MB  |  Pages: {len(screenshots)}")

    finally:
        if server_proc:
            server_proc.kill()


def main():
    parser = argparse.ArgumentParser(description="Export slide deck to PDF")
    parser.add_argument("--output", "-o", default=None, help="Output PDF filename")
    parser.add_argument("--port", "-p", type=int, default=None,
                        help="Server port (default: auto-find available port)")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't auto-start server (expects one already running)")
    args = parser.parse_args()

    deck_dir = os.path.dirname(os.path.abspath(__file__))
    output = args.output or os.path.join(deck_dir, "deck-export.pdf")

    export_slides_to_pdf(
        deck_dir, output,
        port=args.port,
        auto_start=not args.no_server,
    )


if __name__ == "__main__":
    main()
