"""torch.nn: Module, Linear and CrossEntropyLoss, enough for bacpipe's wrappers and its linear probe.

Gradients exist for one case only, a Linear layer followed by CrossEntropyLoss, which is the linear probe.
"""
from collections import OrderedDict

import numpy as np

from .. import Tensor, _as_tensor
from . import functional   # noqa: F401


class Parameter(Tensor):
    grad = None

    def __new__(cls, data):
        return np.array(data, dtype=np.float32).view(cls)

    def __array_finalize__(self, obj):
        self._grad_fn, self.grad = None, None


class Module:
    training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _children(self):
        return [(name, value) for name, value in vars(self).items() if isinstance(value, Module)]

    def named_parameters(self, prefix=""):
        for name, value in vars(self).items():
            if isinstance(value, Parameter):
                yield prefix + name, value
        for name, child in self._children():
            yield from child.named_parameters(prefix + name + ".")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def state_dict(self):
        return OrderedDict((name, p.view(Tensor)) for name, p in self.named_parameters())

    def load_state_dict(self, state_dict, strict=True):
        params = dict(self.named_parameters())
        if strict and set(params) != set(state_dict):
            raise RuntimeError(f"state_dict keys {sorted(state_dict)} do not match the module's {sorted(params)}")
        for name, value in state_dict.items():
            if name in params:
                params[name][...] = np.asarray(value)

    def train(self, mode=True):
        self.training = mode
        for _, child in self._children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        return self


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        bound = 1 / np.sqrt(in_features)                       # PyTorch's default initialisation for Linear
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(np.random.uniform(-bound, bound, (out_features, in_features)))
        self.bias = Parameter(np.random.uniform(-bound, bound, out_features)) if bias else None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        out = x @ np.asarray(self.weight).T
        if self.bias is not None:
            out = out + np.asarray(self.bias)
        out = _as_tensor(out)
        out._grad_fn = (self, x)
        return out


class _Loss:
    def __init__(self, value, backward):
        self._value, self._backward = value, backward

    def item(self):
        return float(self._value)

    def backward(self):
        self._backward()

    def __float__(self):
        return float(self._value)


class CrossEntropyLoss(Module):
    """Mean cross-entropy of logits against class indices."""
    def forward(self, logits, target):
        source = getattr(logits, "_grad_fn", None)
        z = np.asarray(logits, dtype=np.float32)
        y = np.asarray(target).astype(np.int64)
        z = z - z.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        rows = np.arange(len(y))

        def backward():
            if source is None:
                raise RuntimeError("this loss has gradients only for the output of a Linear layer")
            layer, x = source
            d = np.exp(log_p)
            d[rows, y] -= 1
            d /= len(y)
            layer.weight.grad = (d.T @ x).astype(np.float32)
            if layer.bias is not None:
                layer.bias.grad = d.sum(axis=0).astype(np.float32)

        return _Loss(-log_p[rows, y].mean(), backward)
