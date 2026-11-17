# 0001. bacpipe's own code runs in the browser (Pyodide and ONNX Runtime Web)
Status: Accepted
Date: 2026-11-16

## Context

EMBERIZA must be a page on GitHub Pages, like Apus. It must do what bacpipe does, with all of bacpipe's options, and give the same results. The parts of bacpipe 1.3.5 that this needs are about 16,000 lines of Python. Its results are a folder of files (`bacpipe_results/<dataset>/`), and its dashboard is a Panel app that needs a Python server.

GitHub Pages serves static files only. Whatever runs bacpipe's steps must therefore run in the visitor's browser.

## Options considered

1. **bacpipe's own files in Pyodide.** bacpipe's Python is copied unchanged into the page and runs in Pyodide. The networks run in ONNX Runtime Web, from the ONNX files bacpipe itself uses for Perch 2.0 and BirdNET+ V3.0.
   - Gains: the results come from bacpipe's code, so parity is checked rather than rebuilt. A new bacpipe version is a new copy plus the tests. Every option of `config.yaml` and `settings.yaml` reaches the code that reads it.
   - Loses: about 28 MB on the first visit, 15 to 33 s to start, and stand-ins for the libraries Pyodide lacks (ADR 0002).
2. **A JavaScript rewrite of bacpipe's steps.**
   - Gains: lighter and quicker to start.
   - Loses: a second implementation of windowing, labels, predictions, clustering, probing and the file formats, to be kept in step with bacpipe by hand. Parity would have to be built, not checked.
3. **A local app (bacpipe plus a desktop shell).**
   - Gains: the real PyTorch and numba, so no stand-ins.
   - Loses: not a web page, an install of several GB, and it already exists as bacpipe's own dashboard.

## Decision

Option 1.
- **Pinned versions:** Pyodide 314.0.7 (Python 3.14, numpy 2.4.6, scipy 1.18, pandas 3.0.2, scikit-learn 1.8, soundfile, soxr, PyYAML) and ONNX Runtime Web 1.30.0, from jsDelivr.
- **bacpipe:** the 42 files bacpipe 1.3.5 needs, copied from commit `d7aa3a2971ae4287c9f4a10acde324519776296a` by `tools/vendor_bacpipe.py`. `app/py/bacpipe/VENDORED.json` records each file's SHA-256 and `tests/test_vendored.py` checks them.
- **Isolation:** coi-serviceworker gives the page cross-origin isolation, so ONNX Runtime Web can use several threads.
- **Storage:** results are written to the browser's private file system (OPFS), so they open again on the next visit.

## Evidence

[Research 0001](../research/0001-parity-with-bacpipe.md):
- The audio windows bacpipe feeds the networks are identical, on 20 pairs of recording and model.
- With the same ONNX runtime outside the browser, the embeddings are identical to bacpipe's, to the last bit.
- In the browser the embeddings are within 0.000005 of bacpipe's.

## Consequences

- **Easier:** parity, updates of bacpipe, and every option of bacpipe is real.
- **Harder:**
  - Two runtimes share one Web Worker.
  - Only the models with an ONNX file in bacpipe can run (Perch 2.0 and BirdNET+ V3.0). The other 25 need PyTorch or TensorFlow.
  - A recording is read whole, as bacpipe reads it, so very large files may not fit in a tab.
- **Reversal:** none needed for the interface. The page reads and writes bacpipe's own result folders, so a server running the real bacpipe could replace the worker without changing the page.
