"""EMBERIZA's UMAP against umap-learn's own numbers (tests/fixtures/umap_reference.npz, made by
tests/tools/umap_reference.py)."""
from pathlib import Path

import numpy as np
import pytest
from sklearn.manifold import trustworthiness

from emberiza import umap_port as up

REF = np.load(Path(__file__).parent / "fixtures" / "umap_reference.npz")


@pytest.fixture(scope="module")
def fitted():
    return up.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=42).fit(REF["X"])


def test_curve_parameters(fitted):
    assert fitted._a == pytest.approx(float(REF["a"]), rel=1e-9)
    assert fitted._b == pytest.approx(float(REF["b"]), rel=1e-9)


def test_same_graph(fitted):
    graph = fitted.graph_.toarray()
    assert np.array_equal(graph > 0, REF["graph"] > 0), "the port connects different neighbours"
    assert np.abs(graph - REF["graph"]).max() < 1e-5
    assert np.abs(fitted._sigmas - REF["sigmas"]).max() < 1e-5
    assert np.abs(fitted._rhos - REF["rhos"]).max() < 1e-5


def test_update_loop_is_umap_learns_to_the_last_bit():
    """From umap-learn's own start, graph and random state, the port's loop must end where umap-learn's does."""
    moved = up.optimize_layout(REF["start"].copy(), REF["head"], REF["tail"], 500, REF["X"].shape[0], REF["epochs_per_sample"],
                               float(REF["a"]), float(REF["b"]), REF["rng_state"], 1.0, 1.0, 5)
    assert np.array_equal(moved, REF["embedding"])


def test_layout_keeps_neighbours_as_well(fitted):
    """End to end the positions differ (see umap_port.py), so the layouts are compared by what they are for."""
    mine = trustworthiness(REF["X"], fitted.embedding_, n_neighbors=10)
    theirs = trustworthiness(REF["X"], REF["embedding"], n_neighbors=10)
    assert mine > theirs - 0.03, (mine, theirs)


def test_three_dimensions_and_tiny_datasets():
    assert up.UMAP(n_components=3, random_state=42, n_epochs=20).fit_transform(REF["X"]).shape == (60, 3)
    with pytest.warns(UserWarning, match="n_neighbors is larger"):
        assert up.UMAP(random_state=42, n_epochs=20).fit_transform(REF["X"][:8]).shape == (8, 2)


def test_what_is_not_ported_says_so():
    with pytest.raises(NotImplementedError):
        up.UMAP(metric="cosine")
    with pytest.raises(NotImplementedError):
        up.UMAP(random_state=42).transform(REF["X"])
