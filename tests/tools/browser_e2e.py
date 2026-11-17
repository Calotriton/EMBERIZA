"""Drives the real page in Edge (Playwright, with the Edge that is installed), runs bacpipe on a folder of
recordings, downloads the ZIP and compares it with bacpipe's own results. Needs tools/serve.py running.

    python tests/tools/browser_e2e.py <audio folder> <reference>/bacpipe_results/<dataset> [--tasks clustering,probing]
                                      [--models perch_v2,birdnet_v3] [--dim umap] [--headed] [--gpu] [--shots folder]
"""
import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from compare_results import compare, summary  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("audio_dir")
parser.add_argument("reference")
parser.add_argument("--models", default="perch_v2,birdnet_v3")
parser.add_argument("--tasks", default="")
parser.add_argument("--dim", default="umap")
parser.add_argument("--url", default="http://127.0.0.1:8000/app/#fresh")
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "out_browser"))
parser.add_argument("--shots")
parser.add_argument("--headed", action="store_true")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--batch-size", type=int, help="bacpipe's global_batch_size")
args = parser.parse_args()


def shot(page, name):
    if args.shots:
        Path(args.shots).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(args.shots) / f"{name}.png"))


def run_app():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=not args.headed)
        page = browser.new_context(accept_downloads=True, viewport={"width": 1500, "height": 950}).new_page()
        page.on("console", lambda m: print(f"  [console {m.type}] {m.text[:400]}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: print(f"  [page error] {e}"))
        t0 = time.time()
        page.goto(args.url)
        page.wait_for_selector("#engine.ready, #engine.failed", timeout=600_000)
        print(f"ready after {time.time() - t0:.0f} s: {page.inner_text('#about-versions')}")
        if page.locator("#engine.failed").count():
            sys.exit(f"APP DID NOT START: {page.inner_text('#alert')}")
        shot(page, "1-start")
        print("UMAP loop, JavaScript against Python (largest difference):",
              page.evaluate("new Promise((res) => { const w = new Worker('worker.js', {type: 'module'}); w.onmessage = (e) => { if (e.data.type === 'ready') w.postMessage({id: 1, do: 'kernelSelftest'}); else if (e.data.id === 1) { res(e.data.result ?? e.data.error); w.terminate(); } }; })"))
        page.evaluate("localStorage.clear()")
        page.reload()
        page.wait_for_selector("#engine.ready", timeout=600_000)
        for model in ("perch_v2", "birdnet_v3"):
            page.set_checked(f'#models input[value="{model}"]', model in args.models.split(","))
        page.select_option("#opt-dim_reduction_model", args.dim)
        for task in ("clustering", "probing"):
            box = page.locator("#opt-evaluation_task label", has_text=task).locator("input")
            box.set_checked(task in args.tasks.split(","))
        if args.gpu or args.batch_size:
            page.click("details.group >> text=Embedding generation")
        if args.gpu:
            page.select_option("#opt-__backend", "gpu")
        if args.batch_size:
            page.fill("#opt-global_batch_size", str(args.batch_size))
            page.locator("#opt-global_batch_size").dispatch_event("change")
        page.set_input_files("#folder-input", args.audio_dir)
        print("recordings:", page.inner_text("#recordings-facts").replace("\n", " | "))
        shot(page, "2-chosen")
        t1 = time.time()
        page.click("#run")
        page.wait_for_selector("#results:not([hidden]), #alert:not([hidden])", timeout=3_600_000)
        if page.is_visible("#alert"):
            print("LOG TAIL:", page.locator("#log li").all_text_contents()[-12:])
            sys.exit(f"APP ERROR: {page.inner_text('#alert')}")
        print(f"run took {time.time() - t1:.0f} s")
        print("stats:", page.inner_text("#stats").replace("\n", " "))
        print([t for t in page.locator("#log li").all_text_contents() if "runs on" in t or t.startswith("Done") or t.startswith("Timing")])
        page.wait_for_selector("#ex-plot .plot-container", timeout=120_000)
        page.wait_for_timeout(1500)
        shot(page, "3-explore")
        page.evaluate("document.getElementById('ex-plot').emit('plotly_click', {points: [{customdata: 3}]})")
        page.wait_for_selector("#seg-spec .plot-container", timeout=60_000)
        page.wait_for_timeout(800)
        print("window:", page.inner_text("#seg-title"), "| sound:", page.evaluate("document.getElementById('seg-audio').src.slice(0, 5)"),
              "| facts:", page.inner_text("#seg-facts").replace(chr(10), " ")[:200])
        shot(page, "3-explore-window")
        for tab in ("compare", "predictions", "evaluation", "files", "about"):
            if page.is_visible(f"#tab-{tab}"):
                page.click(f"#tab-{tab}")
                page.wait_for_timeout(2500)
                shot(page, f"4-{tab}")
        page.click("#theme")                                     # the light theme redraws every plot
        for tab in ("explore", "compare", "evaluation"):
            if page.is_visible(f"#tab-{tab}"):
                page.click(f"#tab-{tab}")
                page.wait_for_timeout(2000)
                shot(page, f"5-light-{tab}")
        page.click("#tab-files")
        with page.expect_download(timeout=600_000) as download:
            page.click("#download")
        data = Path(download.value.path()).read_bytes()
        browser.close()
    return zipfile.ZipFile(io.BytesIO(data))


z = run_app()
out = Path(args.out)
if out.exists():
    import shutil
    shutil.rmtree(out)
z.extractall(out)
dataset = Path(args.reference).name
print(summary(compare(args.reference, out / "bacpipe_results" / dataset)))
