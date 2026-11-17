"""Makes bacpipe importable where its heavy dependencies cannot be installed (Pyodide, in the browser).

Two kinds of stand-in:
  - working ones in app/py/shims/ (torch, librosa, audioread, tqdm, umap, ...) and emberiza/replay.py (onnxruntime):
    they do what bacpipe needs from the library, and the tests compare bacpipe's results through them with the
    real libraries;
  - empty ones, made here, for the libraries bacpipe only uses to draw (panel, plotly, matplotlib, seaborn) or to
    play sound (sounddevice). EMBERIZA draws and plays in the page, so nothing in them is ever called for a result.

A real library always wins: the stand-ins are found only when the import would otherwise fail.
"""
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path

DRAWING_ONLY = {"panel", "plotly", "matplotlib", "seaborn", "sounddevice", "kaleido"}


class _Nothing:
    """An object that accepts any call, attribute, item or arithmetic and gives back another one like it.

    bacpipe draws in the middle of computing (a confusion matrix is drawn in the function that scores a probe).
    With these, the drawing goes nowhere and the computing carries on.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Nothing()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Nothing()

    def __setattr__(self, name, value):
        pass

    def __getitem__(self, key):
        return _Nothing()

    def __setitem__(self, key, value):
        pass

    def __iter__(self):                 # `fig, ax = plt.subplots()`
        return iter((_Nothing(), _Nothing()))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __bool__(self):
        return False

    def __float__(self):
        return 0.0

    def __int__(self):
        return 0

    __index__ = __int__

    def _same(self, *args):
        return _Nothing()

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _same
    __neg__ = __pos__ = __abs__ = __pow__ = __floordiv__ = __mod__ = _same

    def __repr__(self):
        return "<not available in EMBERIZA: bacpipe draws with this, the page draws instead>"


class _EmptyModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name[:1].isupper():                       # a class, usable in isinstance() and as a base class
            return type(name, (_Nothing,), {"__module__": self.__name__})
        return _Nothing()


class _EmptyFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in DRAWING_ONLY:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _EmptyModule(spec.name)

    def exec_module(self, module):
        pass


def install():
    shims = str(Path(__file__).resolve().parent.parent / "shims")
    if shims not in sys.path:
        sys.path.append(shims)                       # after site-packages: a real library wins
    if not any(isinstance(f, _EmptyFinder) for f in sys.meta_path):
        sys.meta_path.append(_EmptyFinder())
    # The one exception to "a real library wins": bacpipe's `import onnxruntime` always gets the replay session,
    # also in the tests, where the real onnxruntime is installed to play the part of ONNX Runtime Web.
    from . import replay
    sys.modules["onnxruntime"] = replay
