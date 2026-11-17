"""A small stand-in for PyTorch, backed by numpy, so that bacpipe's own code runs where PyTorch cannot be installed
(Pyodide, in the browser). It covers what bacpipe 1.3.5 uses outside its neural networks: tensors as containers,
concatenation, max, softmax, data loaders, and a linear probe trained with Adam. The networks themselves run in
ONNX Runtime Web (see shims/onnxruntime).

Tensors are numpy arrays of a subclass, so numpy functions accept them. As in PyTorch, floating point data defaults
to float32.
"""
from collections import namedtuple

import numpy as np

__version__ = "0+emberiza.numpy"

float32 = float = np.float32      # noqa: A001  (torch.float)
float64 = double = np.float64
int64 = long = np.int64
int32 = np.int32
bool = np.bool_                   # noqa: A001


class device(str):                # noqa: N801  (torch.device)
    """torch.device("cpu"): a string that also has .type."""
    @property
    def type(self):
        return str(self).split(":")[0]


class _Size(int):
    def __new__(cls, array):
        self = int.__new__(cls, np.ndarray.size.__get__(array))
        self._shape = array.shape
        return self

    def __call__(self, dim=None):
        return self._shape if dim is None else self._shape[dim]


class Tensor(np.ndarray):
    """torch.Tensor(data) makes a float32 tensor; use torch.tensor(data) to keep the data's own type."""
    _grad_fn = None

    def __new__(cls, data=()):
        return np.array(data, dtype=np.float32).view(cls)

    def __array_finalize__(self, obj):
        self._grad_fn = None

    # --- no-ops that exist for devices and autograd ---
    def cpu(self):
        return self

    def detach(self):
        return self.view(Tensor)

    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype") or next((a for a in args if isinstance(a, type) and issubclass(a, np.generic)), None)
        return self.astype(dtype).view(Tensor) if dtype is not None else self

    def clone(self):
        return self.copy()

    def numpy(self):
        return np.asarray(self)

    def eval(self):
        return self

    @property
    def device(self):
        return device("cpu")

    # --- shape helpers with PyTorch's names ---
    def dim(self):
        return self.ndim

    @property
    def size(self):
        return _Size(self)        # numpy reads .size as a number, PyTorch calls .size(dim): this is both

    def unsqueeze(self, dim):
        return np.expand_dims(self, dim)

    def squeeze(self, dim=None):
        if dim is not None and self.shape[dim] != 1:      # PyTorch leaves the tensor as it is; numpy raises
            return self
        return np.ndarray.squeeze(self, axis=dim)

    def long(self):
        return self.astype(np.int64)

    def float(self):
        return self.astype(np.float32)

    def item(self):
        return np.ndarray.item(self)


def _as_tensor(array):
    return np.asarray(array).view(Tensor)


def tensor(data, dtype=None, device=None, requires_grad=False):   # noqa: A002
    known = getattr(data, "dtype", None)
    array = np.array(data, dtype=dtype)
    if dtype is None and known is None and array.dtype == np.float64:
        array = array.astype(np.float32)                          # Python floats become float32, as in PyTorch
    return array.view(Tensor)


def as_tensor(data, dtype=None, device=None):   # noqa: A002
    return tensor(data, dtype) if not isinstance(data, np.ndarray) else _as_tensor(data if dtype is None else data.astype(dtype))


def from_numpy(array):
    return _as_tensor(array)


def cat(tensors, dim=0, axis=None):
    """torch.cat. An empty 1-D tensor is skipped, which is how bacpipe starts its running lists."""
    parts = [np.asarray(t) for t in tensors if np.shape(t) != (0,)]
    if not parts:
        return Tensor()
    return np.concatenate(parts, axis=dim if axis is None else axis).view(Tensor)


concat = cat
_MaxResult = namedtuple("max", ["values", "indices"])


def max(x, dim=None, keepdim=False):   # noqa: A001
    x = np.asarray(x)
    if dim is None:
        return _as_tensor(x.max())
    return _MaxResult(_as_tensor(x.max(axis=dim, keepdims=keepdim)), _as_tensor(x.argmax(axis=dim, keepdims=keepdim)))


def softmax(x, dim=-1):
    """Computed in float64 and rounded to the input's type: summing 15,000 float32 terms in a different order
    than PyTorch would otherwise move the result in the sixth decimal."""
    x = np.asarray(x)
    e = np.exp(x.astype(np.float64) - x.max(axis=dim, keepdims=True))
    return (e / e.sum(axis=dim, keepdims=True)).astype(x.dtype, copy=False).view(Tensor)


def sigmoid(x):
    x = np.asarray(x)
    return (1 / (1 + np.exp(-x))).astype(x.dtype, copy=False).view(Tensor)


class no_grad:   # noqa: N801
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, func):
        return func


from . import _serialization, backends, cuda, nn, optim, utils   # noqa: E402
from ._serialization import FloatStorage, LongStorage, load, save   # noqa: E402,F401
