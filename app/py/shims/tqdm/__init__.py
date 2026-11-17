"""tqdm without a terminal: it yields the items and tells a listener how far it is.

Set `tqdm.listener` to a function (description, done, total) to show the progress elsewhere, in the page for example.
"""
import time


class tqdm:   # noqa: N801
    listener = None
    every_s = 0.25

    def __init__(self, iterable=None, desc=None, total=None, **_):
        self.iterable, self.desc, self.n, self._last = iterable, desc or "", 0, 0.0
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        self.total = total

    def _tell(self, force=False):
        now = time.monotonic()
        if tqdm.listener and (force or now - self._last >= tqdm.every_s):
            self._last = now
            tqdm.listener(self.desc, self.n, self.total)

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.n += 1
            self._tell()
        self._tell(force=True)

    def update(self, n=1):
        self.n += n
        self._tell()

    def set_description(self, desc=None, **_):
        self.desc = desc or ""

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def trange(*args, **kwargs):
    return tqdm(range(*args), **kwargs)
