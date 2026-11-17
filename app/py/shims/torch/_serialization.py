"""torch.save and torch.load for dictionaries of tensors, in the file format PyTorch itself writes (a ZIP archive
holding a pickle and the raw numbers), so that a probe trained here opens in PyTorch and the other way round.

Loading is restricted to what such a file needs. Nothing else in a pickle is allowed to run.
"""
import collections
import io
import pickle
import zipfile

import numpy as np


class _Storage:
    dtype = None

    def __init__(self, data=None):
        self.data = data


class FloatStorage(_Storage):
    dtype = np.float32


class LongStorage(_Storage):
    dtype = np.int64


class DoubleStorage(_Storage):
    dtype = np.float64


for _c in (FloatStorage, LongStorage, DoubleStorage):
    _c.__module__ = "torch"          # PyTorch files name them torch.FloatStorage, and so must these

_STORAGES = {np.dtype(c.dtype): c for c in (FloatStorage, LongStorage, DoubleStorage)}


def save(obj, f):
    from . import _utils
    arrays = []

    class Pickler(pickle.Pickler):
        def persistent_id(self, item):
            if isinstance(item, _Storage):
                arrays.append(item.data)
                return ("storage", type(item), str(len(arrays) - 1), "cpu", item.data.size)
            return None

        def reducer_override(self, item):
            if isinstance(item, np.ndarray):
                data = np.ascontiguousarray(item)
                if data.dtype not in _STORAGES:
                    data = data.astype(np.float32)
                strides = tuple(int(np.prod(data.shape[i + 1:])) for i in range(data.ndim))
                return (_utils._rebuild_tensor_v2,
                        (_STORAGES[data.dtype](data.reshape(-1)), 0, tuple(data.shape), strides, False, collections.OrderedDict()))
            return NotImplemented

    buffer = io.BytesIO()
    Pickler(buffer, protocol=2).dump(obj)
    target = f if hasattr(f, "write") else open(f, "wb")
    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as z:
            z.writestr("archive/data.pkl", buffer.getvalue())
            z.writestr("archive/byteorder", "little")
            for i, data in enumerate(arrays):
                z.writestr(f"archive/data/{i}", data.astype(data.dtype.newbyteorder("<"), copy=False).tobytes())
            z.writestr("archive/version", "3\n")
    finally:
        if target is not f:
            target.close()


def load(f, map_location=None, weights_only=True, **_):
    from . import _utils
    allowed = {("collections", "OrderedDict"): collections.OrderedDict,
               ("torch._utils", "_rebuild_tensor_v2"): _utils._rebuild_tensor_v2,
               ("torch", "FloatStorage"): FloatStorage, ("torch", "LongStorage"): LongStorage,
               ("torch", "DoubleStorage"): DoubleStorage}
    with zipfile.ZipFile(f) as z:
        root = next(n for n in z.namelist() if n.endswith("data.pkl"))[: -len("data.pkl")]

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) not in allowed:
                    raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a file of tensors")
                return allowed[(module, name)]

            def persistent_load(self, pid):
                _, storage_class, key, _, _ = pid
                return storage_class(np.frombuffer(z.read(f"{root}data/{key}"), dtype=np.dtype(storage_class.dtype).newbyteorder("<")))

        return Unpickler(io.BytesIO(z.read(root + "data.pkl"))).load()
