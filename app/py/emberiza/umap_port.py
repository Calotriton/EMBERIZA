"""UMAP for where umap-learn cannot be installed: it needs numba, which has no build for the browser.

This is a port of the part of umap-learn 0.5.12 that bacpipe uses,

    umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2 or 3, metric="euclidean", random_state=42).fit_transform(X)

following its source step by step (umap/umap_.py, umap/spectral.py, umap/layouts.py, umap/utils.py): the fuzzy graph,
the spectral start, the curve parameters, the random numbers (numpy's seeded generator for the start, umap's own
Tausworthe generator for the negative samples) and the order of the updates. From umap-learn's own start, the update
loop ends at umap-learn's positions to the last bit. End to end the positions differ: the graph's weights differ in
their last digits, the eigenvectors of the start come from another SciPy (another build of ARPACK), and umap-learn
seeds each point's random numbers from the bits of its start position, so a start that differs in the last bit takes
another path. tests/test_umap.py compares the two.

Two differences, both stated here because they change what is computed:
  - the nearest neighbours are always exact. umap-learn computes them exactly below 4,096 points and approximately
    (NN-descent) above;
  - only the euclidean metric, unsupervised, without densMAP, and no `transform` of new points.

umap-learn is by Leland McInnes and contributors, BSD 3-Clause licence:
Copyright (c) 2017, Leland McInnes. All rights reserved. See LICENSES.md.
"""
import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from scipy.optimize import curve_fit
from sklearn.utils import check_array, check_random_state

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_FLOATMAX = np.finfo(np.float32).max

# The page sets these: the update loop in JavaScript (the same loop as optimize_layout below, far faster than
# Python without numba), and a function (done, total) that shows the progress.
layout_kernel = None
progress = None


# ----------------------------------------------------------------------------- the graph

def exact_knn(X, k, chunk=512):
    """The k nearest neighbours of every row, itself first: indices (n, k) int32 and distances (n, k) float32.

    The neighbours are found from squared distances computed with one matrix product per chunk. Their distances
    are then computed the way umap-learn's euclidean does (differences in float32, summed in float64), and the k
    are ordered by distance, ties by index, which is the order of umap-learn's stable sort.
    """
    n = X.shape[0]
    X64 = X.astype(np.float64)
    sq = np.einsum("ij,ij->i", X64, X64)
    indices = np.empty((n, k), dtype=np.int32)
    dists = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * (X64[start:stop] @ X64.T)
        d2[np.arange(stop - start), np.arange(start, stop)] = -1.0                  # itself always comes first
        if k < n:
            cand = np.argpartition(d2, k - 1, axis=1)[:, :k]
        else:
            cand = np.tile(np.arange(n), (stop - start, 1))
        diff = X[cand] - X[start:stop, None, :]                                      # float32, as in umap-learn
        exact = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff, dtype=np.float64))
        order = np.lexsort((cand, exact), axis=1)
        indices[start:stop] = np.take_along_axis(cand, order, axis=1)
        dists[start:stop] = np.take_along_axis(exact, order, axis=1)
        if progress:
            progress(stop, n)
    return indices, dists.astype(np.float32)


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """umap_.smooth_knn_dist, for all rows at once: sigma and rho of every point."""
    target = np.log2(k) * bandwidth
    n = distances.shape[0]
    rho = np.zeros(n, dtype=np.float32)
    mean_distances = np.mean(distances)
    index = int(np.floor(local_connectivity))
    interpolation = local_connectivity - index
    for i in range(n):
        non_zero = distances[i][distances[i] > 0.0]
        if non_zero.shape[0] >= local_connectivity:
            if index > 0:
                rho[i] = non_zero[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (non_zero[index] - non_zero[index - 1])
            else:
                rho[i] = interpolation * non_zero[0]
        elif non_zero.shape[0] > 0:
            rho[i] = np.max(non_zero)

    d = (distances[:, 1:] - rho[:, None]).astype(np.float64)        # the subtraction in float32, as in umap-learn
    positive = d > 0
    lo = np.zeros(n)
    hi = np.full(n, float(NPY_FLOATMAX))
    mid = np.ones(n)
    active = np.ones(n, dtype=bool)
    for _ in range(n_iter):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            psum = np.where(positive, np.exp(-(d / mid[:, None])), 1.0).sum(axis=1)
        active &= ~(np.fabs(psum - target) < SMOOTH_K_TOLERANCE)
        if not active.any():
            break
        above = active & (psum > target)
        below = active & ~(psum > target)
        hi[above] = mid[above]
        mid[above] = (lo[above] + hi[above]) / 2.0
        lo[below] = mid[below]
        unbounded = below & (hi >= NPY_FLOATMAX)
        bounded = below & ~(hi >= NPY_FLOATMAX)
        mid[unbounded] *= 2
        mid[bounded] = (lo[bounded] + hi[bounded]) / 2.0
    result = mid.astype(np.float32)
    floor_own = MIN_K_DIST_SCALE * distances.mean(axis=1)
    floor_all = MIN_K_DIST_SCALE * mean_distances
    own = rho > 0.0
    result[own] = np.maximum(result[own], floor_own[own].astype(np.float32))
    result[~own] = np.maximum(result[~own], np.float32(floor_all))
    return result, rho


def fuzzy_simplicial_set(knn_indices, knn_dists, n_neighbors, set_op_mix_ratio=1.0, local_connectivity=1.0):
    """umap_.fuzzy_simplicial_set with compute_membership_strengths: the weighted graph UMAP lays out."""
    n = knn_indices.shape[0]
    knn_dists = knn_dists.astype(np.float32)
    sigmas, rhos = smooth_knn_dist(knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity))
    rows = np.repeat(np.arange(n, dtype=np.int32), knn_indices.shape[1])
    cols = knn_indices.reshape(-1).astype(np.int32)
    shifted = knn_dists - rhos[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.exp(-(shifted / sigmas[:, None])).astype(np.float32)
    vals[(shifted <= 0.0) | (sigmas[:, None] == 0.0)] = 1.0
    vals[knn_indices == np.arange(n)[:, None]] = 0.0
    result = scipy.sparse.coo_matrix((vals.reshape(-1), (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = set_op_mix_ratio * (result + transpose - prod_matrix) + (1.0 - set_op_mix_ratio) * prod_matrix
    result.eliminate_zeros()
    return result, sigmas, rhos


def find_ab_params(spread, min_dist):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


# ----------------------------------------------------------------------------- the start

def _component_layout(data, n_components, component_labels, dim, random_state):
    from sklearn.manifold import SpectralEmbedding
    from sklearn.metrics import pairwise_distances
    if data is None:
        return np.random.random(size=(n_components, dim)) * 10.0
    centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)
    for label in range(n_components):
        centroids[label] = data[component_labels == label].mean(axis=0)
    distance_matrix = pairwise_distances(centroids, metric="euclidean")
    affinity_matrix = np.exp(-(distance_matrix ** 2))
    embedding = SpectralEmbedding(n_components=dim, affinity="precomputed", random_state=random_state).fit_transform(affinity_matrix)
    embedding /= embedding.max()
    return embedding


def _multi_component_layout(data, graph, n_components, component_labels, dim, random_state):
    from sklearn.metrics import pairwise_distances
    result = np.empty((graph.shape[0], dim), dtype=np.float32)
    if n_components > 2 * dim:
        meta_embedding = _component_layout(data, n_components, component_labels, dim, random_state)
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]
    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()
        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0
        if component_graph.shape[0] < 2 * dim or component_graph.shape[0] <= dim + 1:
            result[component_labels == label] = (
                random_state.uniform(low=-data_range, high=data_range, size=(component_graph.shape[0], dim))
                + meta_embedding[label])
        else:
            component_embedding = spectral_layout(None, component_graph, dim, random_state)
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = component_embedding + meta_embedding[label]
    return result


def spectral_layout(data, graph, dim, random_state):
    """spectral._spectral_layout: the eigenvectors of the graph's normalised Laplacian."""
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)
    if n_components > 1:
        return _multi_component_layout(data, graph, n_components, labels, dim, random_state)
    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    identity = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(1.0 / sqrt_deg, 0, graph.shape[0], graph.shape[0])
    L = identity - D * graph * D
    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        random_state.normal(size=(L.shape[0], k))       # umap-learn draws these and does not use them with eigsh;
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(     # drawn here too, to stay on the same random numbers
            L, k, which="SM", ncv=num_lanczos_vectors, tol=1e-4, v0=np.ones(L.shape[0]), maxiter=graph.shape[0] * 5)
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except scipy.sparse.linalg.ArpackError:
        warnings.warn("Spectral initialisation failed! The eigenvector solver failed. This is likely due to too "
                      "small an eigengap. Falling back to random initialisation!")
        return random_state.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))


def noisy_scale_coords(coords, random_state, max_coord=10.0, noise=0.0001):
    expansion = max_coord / np.abs(coords).max()
    coords = (coords * expansion).astype(np.float32)
    return coords + random_state.normal(scale=noise, size=coords.shape).astype(np.float32)


# ----------------------------------------------------------------------------- the layout

def make_epochs_per_sample(weights, n_epochs):
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


def _tau_rand_int(state):
    """utils.tau_rand_int on a list of three Python integers that hold 64-bit values, returned as a signed int32."""
    s0, s1, s2 = state
    s0 = (((s0 & 4294967294) << 12) & 0xFFFFFFFF) ^ ((((s0 << 13) & 0xFFFFFFFF) ^ s0) >> 19)
    s1 = (((s1 & 4294967288) << 4) & 0xFFFFFFFF) ^ ((((s1 << 2) & 0xFFFFFFFF) ^ s1) >> 25)
    s2 = (((s2 & 4294967280) << 17) & 0xFFFFFFFF) ^ ((((s2 << 3) & 0xFFFFFFFF) ^ s2) >> 11)
    state[0], state[1], state[2] = s0, s1, s2
    r = (s0 ^ s1 ^ s2) & 0xFFFFFFFF
    return r - 0x100000000 if r >= 0x80000000 else r


def optimize_layout(embedding, head, tail, n_epochs, n_vertices, epochs_per_sample, a, b, rng_state,
                    gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0):
    """layouts.optimize_layout_euclidean with move_other=True, one edge after the other (the order umap-learn uses
    when a random_state is given). `embedding` is float32 and is changed in place.

    This Python loop is the reference. In the page, `layout_kernel` runs the same loop in JavaScript.

    The precision of each step is umap-learn's: positions are float32, the squared distance is summed in float32,
    the gradient is float64, and each new position is rounded to float32. It matters. The repulsion between two
    close points is clipped at 4 units in a direction set by the sign of a tiny difference, so a change in the
    last digit grows to whole units within a few epochs. With every step at the same precision, the layout is
    umap-learn's.
    """
    f32 = np.float32
    zero = np.float32(0.0)
    dim = embedding.shape[1]
    states = (np.full((embedding.shape[0], 3), rng_state, dtype=np.int64)
              + embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1))
    if layout_kernel is not None:
        return layout_kernel(embedding, head, tail, int(n_epochs), int(n_vertices), epochs_per_sample, float(a), float(b),
                             states, float(gamma), float(initial_alpha), float(negative_sample_rate))
    states = states.tolist()
    head, tail = head.tolist(), tail.tolist()
    emb = [[f32(v) for v in row] for row in embedding]
    eps = epochs_per_sample.tolist()
    eps_neg = (epochs_per_sample / negative_sample_rate).tolist()
    next_neg = list(eps_neg)
    next_sample = list(eps)
    alpha = initial_alpha
    dims = range(dim)
    for n in range(n_epochs):
        for i in range(len(eps)):
            if next_sample[i] <= n:
                j, k = head[i], tail[i]
                current, other = emb[j], emb[k]
                dist_squared = zero
                for d in dims:
                    diff = float(current[d] - other[d])
                    dist_squared = f32(float(dist_squared) + diff * diff)    # float32, fused: see the note above
                dist_squared = float(dist_squared)
                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                    grad_coeff /= a * pow(dist_squared, b) + 1.0
                else:
                    grad_coeff = 0.0
                for d in dims:
                    grad_d = grad_coeff * float(current[d] - other[d])
                    grad_d = 4.0 if grad_d > 4.0 else (-4.0 if grad_d < -4.0 else grad_d)
                    current[d] = f32(float(current[d]) + grad_d * alpha)
                    other[d] = f32(float(other[d]) + -grad_d * alpha)
                next_sample[i] += eps[i]
                n_neg_samples = int((n - next_neg[i]) / eps_neg[i])
                for _ in range(n_neg_samples):
                    k = _tau_rand_int(states[j]) % n_vertices
                    other = emb[k]
                    dist_squared = zero
                    for d in dims:
                        diff = float(current[d] - other[d])
                        dist_squared = f32(float(dist_squared) + diff * diff)
                    dist_squared = float(dist_squared)
                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0
                    for d in dims:
                        if grad_coeff > 0.0:
                            grad_d = grad_coeff * float(current[d] - other[d])
                            grad_d = 4.0 if grad_d > 4.0 else (-4.0 if grad_d < -4.0 else grad_d)
                        else:
                            grad_d = 0.0
                        current[d] = f32(float(current[d]) + grad_d * alpha)
                next_neg[i] += n_neg_samples * eps_neg[i]
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
        if progress and n % 10 == 0:
            progress(n + 1, n_epochs)
    embedding[:] = np.array(emb, dtype=np.float32)
    return embedding


def simplicial_set_embedding(data, graph, n_components, initial_alpha, a, b, gamma, negative_sample_rate, n_epochs,
                             random_state):
    """umap_.simplicial_set_embedding with init="spectral" and the euclidean output metric."""
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    default_epochs = 500 if graph.shape[0] <= 10000 else 200
    if n_epochs is None:
        n_epochs = default_epochs
    if n_epochs > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0
    graph.eliminate_zeros()
    embedding = spectral_layout(data, graph, n_components, random_state)
    embedding = noisy_scale_coords(embedding, random_state, max_coord=10, noise=0.0001)
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
    head, tail = graph.row, graph.col
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    embedding = (10.0 * (embedding - np.min(embedding, 0))
                 / (np.max(embedding, 0) - np.min(embedding, 0))).astype(np.float32, order="C")
    return optimize_layout(embedding, head, tail, n_epochs, n_vertices, epochs_per_sample, a, b, rng_state,
                           gamma, initial_alpha, negative_sample_rate)


class UMAP:
    def __init__(self, n_neighbors=15, n_components=2, metric="euclidean", n_epochs=None, learning_rate=1.0,
                 min_dist=0.1, spread=1.0, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0,
                 negative_sample_rate=5, random_state=None, **other):
        if metric != "euclidean":
            raise NotImplementedError(f"metric {metric!r}: only the euclidean metric is available here")
        if other:
            raise NotImplementedError(f"these UMAP options are not available here: {sorted(other)}")
        self.n_neighbors, self.n_components, self.metric, self.n_epochs = n_neighbors, n_components, metric, n_epochs
        self.learning_rate, self.min_dist, self.spread = learning_rate, min_dist, spread
        self.set_op_mix_ratio, self.local_connectivity = set_op_mix_ratio, local_connectivity
        self.repulsion_strength, self.negative_sample_rate, self.random_state = repulsion_strength, negative_sample_rate, random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float32, order="C")
        self._a, self._b = find_ab_params(self.spread, self.min_dist)
        n = X.shape[0]
        if n <= self.n_neighbors:
            if n == 1:
                self.embedding_ = np.zeros((1, self.n_components))
                return self
            warnings.warn("n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1")
            self._n_neighbors = n - 1
        else:
            self._n_neighbors = self.n_neighbors
        random_state = check_random_state(self.random_state)
        self._knn_indices, self._knn_dists = exact_knn(X, self._n_neighbors)
        self.graph_, self._sigmas, self._rhos = fuzzy_simplicial_set(
            self._knn_indices, self._knn_dists, self._n_neighbors, self.set_op_mix_ratio, self.local_connectivity)
        self.embedding_ = simplicial_set_embedding(
            X, self.graph_, self.n_components, self.learning_rate, self._a, self._b, self.repulsion_strength,
            self.negative_sample_rate, self.n_epochs, random_state)
        disconnected = np.array(self.graph_.sum(axis=1)).flatten() == 0
        if disconnected.any():
            self.embedding_[disconnected] = np.full(self.n_components, np.nan)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_

    def transform(self, X):
        raise NotImplementedError("placing new points in an existing UMAP is not available here")
