"""librosa.util.fix_length, as in librosa 0.11."""
import numpy as np


def fix_length(data, *, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]
    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]
    if n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)
    return data
