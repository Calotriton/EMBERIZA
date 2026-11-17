"""torch.optim.Adam with PyTorch's defaults."""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.betas, self.eps, self.t = list(params), lr, betas, eps, 0
        self.m = [np.zeros(p.shape, np.float32) for p in self.params]
        self.v = [np.zeros(p.shape, np.float32) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m *= b1
            m += (1 - b1) * p.grad
            v *= b2
            v += (1 - b2) * p.grad * p.grad
            step = self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + self.eps)
            p -= step.astype(np.float32)
