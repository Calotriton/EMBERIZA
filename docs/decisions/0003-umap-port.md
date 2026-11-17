# 0003. UMAP is a port of umap-learn, with its update loop also in JavaScript
Status: Accepted
Date: 2026-11-16

## Context

UMAP is bacpipe's default map (`dim_reduction_model: umap`). bacpipe calls umap-learn 0.5.12 with fixed parameters:

    umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2 or 3, metric="euclidean", random_state=42)

umap-learn needs numba, and numba has no build for Pyodide. Without UMAP the default of bacpipe would not work in the page.

## Options considered

1. **Port the part of umap-learn that bacpipe uses**, step by step from its source: the exact nearest neighbours, `smooth_knn_dist`, the fuzzy set, the spectral start (`eigsh`), `noisy_scale_coords`, `make_epochs_per_sample`, umap-learn's Tausworthe generator and its update loop.
   - Gains: the same method and the same random numbers, checked against umap-learn's own numbers.
   - Loses: about 400 lines to keep in step with umap-learn. The update loop is slow in plain Python.
2. **A JavaScript UMAP library** (umap-js).
   - Loses: another algorithm in its details (approximate neighbours, another generator, another start), so another map.
3. **Offer PCA and t-SNE only.**
   - Loses: bacpipe's default.

## Decision

Option 1, in `app/py/emberiza/umap_port.py`, installed as the module `umap`.
- **Positions in float32**, and each squared distance rounded as the compiled loop rounds it. This is what makes the loop match umap-learn's to the last bit.
- **The update loop also exists in JavaScript** (`layoutKernel` in `app/worker.js`), with the 64-bit generator in two 32-bit halves. The page uses it. Python keeps the reference loop, and the page checks at start-up that both give the same positions (the e2e test prints the largest difference, which is 0).
- **What is not ported** raises an error rather than doing something else: other metrics, supervised UMAP, densMAP and `transform`.
- **Neighbours are always exact.** umap-learn uses exact neighbours below 4,096 points and approximate ones above. Above that size the port is slower than umap-learn but still exact.

## Evidence

`tests/test_umap.py` with umap-learn's own numbers (`tests/fixtures/umap_reference.npz`) and [Research 0001](../research/0001-parity-with-bacpipe.md):
- The graph has the same edges, with weights within 0.00001.
- From umap-learn's own start, graph and random state, the port's loop ends at umap-learn's positions to the last bit.
- End to end the positions differ. The start comes from `eigsh`, and SciPy differs (1.18 in Pyodide, 1.17 in bacpipe's environment). The graph's weights differ in their last digits. umap-learn seeds each point's random numbers from the bits of its start position, so a start that differs in the last bit takes another path.
- The maps keep neighbours as well as umap-learn's: trustworthiness within 0.02 of bacpipe's on both test sets.

## Consequences

- **Easier:** bacpipe's default map works in the page, and the result files have bacpipe's layout.
- **Harder:** a UMAP map is not the same picture as bacpipe's for the same recordings. The README says so.
- **Reversal:** if numba or umap-learn becomes available in Pyodide, the stand-in module is removed and the real one is used. The runtime already prefers a real library over its stand-in.
