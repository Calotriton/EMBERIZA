"""Makes tests/fixtures/umap_reference.npz: what umap-learn itself computes on a small made-up dataset, for
tests/test_umap.py to compare EMBERIZA's port with. Run it with an environment that has umap-learn 0.5.12
(bacpipe's own, for example):

    <bacpipe>/.venv/Scripts/python tests/tools/umap_reference.py

The data are random numbers from a fixed seed, so no recording is involved.
"""
from pathlib import Path

import numpy as np
import umap
import umap.layouts as ul
import umap.spectral as us
import umap.umap_ as uu
from sklearn.utils import check_random_state

rng = np.random.RandomState(7)
X = np.vstack([c + rng.normal(size=(15, 16)) for c in rng.normal(size=(4, 16)) * 5]).astype(np.float32)
model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=42).fit(X)

# The pieces of the layout, taken apart the way umap-learn puts them together
rs = check_random_state(42)
graph = model.graph_.tocoo()
graph.sum_duplicates()
graph.data[graph.data < (graph.data.max() / 500.0)] = 0.0
graph.eliminate_zeros()
start = us.spectral_layout(X, graph, 2, rs)
start = uu.noisy_scale_coords(start, rs, max_coord=10, noise=0.0001)
rng_state = rs.randint(uu.INT32_MIN, uu.INT32_MAX, 3).astype(np.int64)
start = (10.0 * (start - np.min(start, 0)) / (np.max(start, 0) - np.min(start, 0))).astype(np.float32, order="C")
epochs_per_sample = uu.make_epochs_per_sample(graph.data, 500)
moved = start.copy()
ul.optimize_layout_euclidean(moved, moved, graph.row, graph.col, 500, graph.shape[1], epochs_per_sample, model._a, model._b,
                             rng_state, 1.0, 1.0, 5, parallel=False, move_other=True)
assert np.array_equal(moved, model.embedding_), "umap-learn's pieces do not add up to its own result"

out = Path(__file__).resolve().parents[1] / "fixtures" / "umap_reference.npz"
np.savez_compressed(out, X=X, graph=model.graph_.toarray(), sigmas=model._sigmas, rhos=model._rhos, a=model._a, b=model._b,
                    start=start, rng_state=rng_state, head=graph.row, tail=graph.col, epochs_per_sample=epochs_per_sample,
                    embedding=model.embedding_, umap_version=umap.__version__)
print(out, X.shape, "edges", len(graph.row))
