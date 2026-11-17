"""The function that PyTorch files name to rebuild a tensor. It must live at torch._utils for the files to match."""
import numpy as np


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    from . import _as_tensor
    count = int(np.prod(size)) if len(size) else 1
    return _as_tensor(np.array(storage.data[storage_offset:storage_offset + count]).reshape(tuple(size)))
