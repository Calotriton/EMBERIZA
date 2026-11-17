# 0002. The loop over recordings is cut around the network, and stand-ins replace what Pyodide lacks
Status: Accepted
Date: 2026-11-16

## Context

ADR 0001 runs bacpipe's files unchanged in Pyodide. Two things stop them from running as they are.

1. **The network.** bacpipe calls `onnxruntime.InferenceSession.run` inside its loop over recordings. In the browser the network runs in ONNX Runtime Web, which answers asynchronously. Python in Pyodide cannot wait for it in the middle of a call.
2. **Missing libraries.** bacpipe imports PyTorch, librosa, audioread, tqdm, umap-learn, huggingface_hub, ijson, Panel, Plotly, matplotlib, seaborn and sounddevice. Pyodide has none of them.

## Options considered

For the network:
1. **Cut the loop around the network.** `app/py/emberiza/engine.py` makes the calls `bacpipe.play()` makes, in the same order. For each recording, bacpipe's own code prepares the windows (`prepare`). The worker runs the network on them. A stand-in `InferenceSession` (`emberiza/replay.py`) is then filled with the outputs, and bacpipe's own `get_embeddings_for_audio`, saving and classifier code run on them (`store`).
   - Gains: bacpipe's code is not changed. The order of windows and batches is bacpipe's.
   - Loses: the engine must follow `play()` and `run_pipeline_for_single_model` closely, and is checked against them by the tests.
2. **Edit bacpipe's files** to make the loop asynchronous.
   - Loses: the copy is no longer bacpipe's, and every update needs the edits again.

For the libraries:
1. **Small stand-ins** in `app/py/shims/` that do only what bacpipe needs:
   - PyTorch: tensors on numpy, `nn.Linear`, `CrossEntropyLoss`, Adam, `DataLoader`, and `torch.save` and `torch.load` in PyTorch's own file format. These carry bacpipe's linear probe.
   - librosa: `load`, `get_duration`, `stft` and `amplitude_to_db`, with soundfile and soxr, the libraries librosa itself calls.
   - audioread, tqdm, huggingface_hub, ijson: what bacpipe reads from them.
   - umap-learn: the port of ADR 0004.
   - Drawing libraries (Panel, Plotly, matplotlib, seaborn, sounddevice, kaleido): modules that accept any call and do nothing (`emberiza/runtime.py`). Six bacpipe functions that only draw PNG files are replaced by functions that do nothing.
2. **Build the real libraries for Pyodide.**
   - Loses: PyTorch and numba have no Pyodide build, and building them is far outside this project.

## Decision

Option 1 for both. A real library always wins over its stand-in when it is installed, except onnxruntime, which is always the replay session. This lets the same engine run outside the browser with the real libraries, which is how `tests/test_parity.py` checks it.

The page draws the plots itself from bacpipe's result files (Plotly.js), as bacpipe's dashboard draws them from the same files. bacpipe's PNG files are not made.

## Evidence

[Research 0001](../research/0001-parity-with-bacpipe.md):
- With the real onnxruntime behind the replay session, the embeddings, `metadata.yml`, the label tables and BirdNET's predictions are identical to bacpipe's.
- A probe saved by the stand-in `torch.save` loads in the real PyTorch, and the reverse.
- The kNN probe gives bacpipe's scores. The linear probe and k-means start from unseeded random numbers in bacpipe, so they differ from run to run in bacpipe too.

## Consequences

- **Easier:** bacpipe stays unchanged. Updating it is `tools/vendor_bacpipe.py`, `tools/make_manifest.py` and the tests.
- **Harder:**
  - A new bacpipe version that uses more of PyTorch or librosa needs more in the stand-ins. `tests/test_standins.py` and the parity test show where.
  - The engine must follow `play()` by hand.
- **Reversal:** if bacpipe one day exposes its per-recording loop as a step that can be resumed, the engine can call it directly.
