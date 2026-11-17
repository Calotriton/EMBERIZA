"""torch.utils.data: a Dataset base class, and a DataLoader that batches in order or shuffled."""
import numpy as np

from .. import Tensor, _as_tensor


class Dataset:
    pass


def _collate(items):
    first = items[0]
    if isinstance(first, (tuple, list)):
        return [_collate([item[i] for item in items]) for i in range(len(first))]
    stacked = np.stack([np.asarray(item) for item in items])
    if stacked.dtype == np.float64 and not isinstance(first, np.ndarray):
        stacked = stacked.astype(np.float32)
    return _as_tensor(stacked)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, **_):
        self.dataset, self.batch_size, self.shuffle, self.drop_last = dataset, batch_size, shuffle, drop_last

    def __len__(self):
        full, rest = divmod(len(self.dataset), self.batch_size)
        return full + (1 if rest and not self.drop_last else 0)

    def __iter__(self):
        n = len(self.dataset)
        if isinstance(self.dataset, np.ndarray) and not self.shuffle:      # a tensor: slices, without copying
            for start in range(0, len(self) * self.batch_size, self.batch_size):
                yield self.dataset[start:start + self.batch_size].view(Tensor)
            return
        order = np.random.permutation(n) if self.shuffle else np.arange(n)
        for b in range(len(self)):
            yield _collate([self.dataset[int(i)] for i in order[b * self.batch_size:(b + 1) * self.batch_size]])
