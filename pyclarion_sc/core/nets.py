"""
Small neural networks for Clarion's implicit (bottom) levels, in plain NumPy.

- MLP:  backpropagation network (used for ACS bottom-level Q-values and
        implicit attitudes).
- LSTM: single-layer LSTM with a dense output layer (used by the NACS as a
        working memory, as in Gomez & Sun, 2024).

No deep-learning framework is needed, so the package runs on small robot
computers too. Both networks train with Adam and full backpropagation
(backpropagation through time for the LSTM).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


_ACT = {
    "relu": (lambda x: np.maximum(x, 0.0), lambda y: (y > 0).astype(float)),
    "tanh": (np.tanh, lambda y: 1.0 - y * y),
    "sigmoid": (_sigmoid, lambda y: y * (1.0 - y)),
    "linear": (lambda x: x, lambda y: np.ones_like(y)),
}


class _Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# --------------------------------------------------------------------------
# MLP (backpropagation network)
# --------------------------------------------------------------------------

@dataclass
class MLP:
    """
    Fully connected backpropagation network.

    sizes:  layer sizes, e.g. (4, 8, 4) = 4 inputs, 8 hidden, 4 outputs.
    hidden: activation for hidden layers ("relu", "tanh", "sigmoid").
    output: activation for the output layer ("linear", "sigmoid").
    """
    sizes: tuple[int, ...]
    hidden: str = "relu"
    output: str = "linear"
    seed: int = 0
    W: list = field(default_factory=list, repr=False)
    b: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self.W:
            rng = np.random.default_rng(self.seed)
            for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
                scale = np.sqrt(2.0 / n_in) if self.hidden == "relu" else np.sqrt(1.0 / n_in)
                self.W.append(rng.normal(0.0, scale, (n_in, n_out)))
                self.b.append(np.zeros(n_out))

    def _forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            name = self.output if i == len(self.W) - 1 else self.hidden
            acts.append(_ACT[name][0](acts[-1] @ W + b))
        return acts

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._forward(X)[-1]

    def fit(self, X, Y, epochs=500, lr=1e-2, batch_size=None, seed=0, l2=0.0) -> list[float]:
        """Train with mean-squared error. Returns the loss history."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        rng = np.random.default_rng(seed)
        opt = _Adam(self.W + self.b, lr=lr)
        n = len(X)
        bs = n if batch_size is None else batch_size
        history = []
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, bs):
                idx = order[s:s + bs]
                acts = self._forward(X[idx])
                err = acts[-1] - Y[idx]
                delta = 2.0 * err / len(idx) * _ACT[self.output][1](acts[-1])
                gW, gb = [], []
                for i in range(len(self.W) - 1, -1, -1):
                    gW.insert(0, acts[i].T @ delta + l2 * self.W[i])
                    gb.insert(0, delta.sum(axis=0))
                    if i > 0:
                        delta = (delta @ self.W[i].T) * _ACT[self.hidden][1](acts[i])
                opt.step(gW + gb)
            history.append(float(np.mean((self.predict(X) - Y) ** 2)))
        return history

    # persistence -----------------------------------------------------------
    def state_dict(self) -> dict:
        d = {"sizes": np.array(self.sizes), "hidden": self.hidden, "output": self.output}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"], d[f"b{i}"] = W, b
        return d


# --------------------------------------------------------------------------
# LSTM (working memory in the NACS)
# --------------------------------------------------------------------------

class LSTM:
    """
    Single-layer LSTM (Hochreiter & Schmidhuber, 1997) + dense sigmoid output.

    Input  x_t: (batch, n_in);  output y_t: (batch, n_out) in [0, 1].
    Gate order in the stacked weight matrix: input, forget, output, candidate.
    """

    def __init__(self, n_in: int, n_hidden: int = 64, n_out: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        k = 1.0 / np.sqrt(n_hidden)
        self.Wx = rng.uniform(-k, k, (n_in, 4 * n_hidden))
        self.Wh = rng.uniform(-k, k, (n_hidden, 4 * n_hidden))
        self.b = np.zeros(4 * n_hidden)
        self.b[n_hidden:2 * n_hidden] = 1.0  # forget-gate bias: remember by default
        self.Wy = rng.uniform(-k, k, (n_hidden, n_out))
        self.by = np.zeros(n_out)

    @property
    def params(self):
        return [self.Wx, self.Wh, self.b, self.Wy, self.by]

    def forward(self, X):
        """X: (T, batch, n_in). Returns output logits (T, batch, n_out) and a cache."""
        T, B, _ = X.shape
        H = self.n_hidden
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache, logits = [], np.zeros((T, B, self.n_out))
        for t in range(T):
            z = X[t] @ self.Wx + h @ self.Wh + self.b
            i = _sigmoid(z[:, :H])
            f = _sigmoid(z[:, H:2 * H])
            o = _sigmoid(z[:, 2 * H:3 * H])
            g = np.tanh(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((X[t], h, c, i, f, o, g, c_new, tc, h_new))
            h, c = h_new, c_new
            logits[t] = h @ self.Wy + self.by
        return logits, cache

    def predict_proba(self, X):
        logits, _ = self.forward(X)
        return _sigmoid(logits)

    def fit(self, sampler, steps=2000, lr=5e-3, seed=0, clip=5.0, verbose=False) -> list[float]:
        """
        sampler(rng) -> (X, Y, M): inputs (T,B,n_in), binary targets (T,B,n_out),
        mask (T,B) marking time steps that are scored. Loss: masked binary
        cross-entropy. Returns the loss history.
        """
        rng = np.random.default_rng(seed)
        opt = _Adam(self.params, lr=lr)
        H = self.n_hidden
        history = []
        for step in range(steps):
            X, Y, M = sampler(rng)
            logits, cache = self.forward(X)
            P = _sigmoid(logits)
            norm = max(M.sum(), 1.0)
            loss = -np.sum(M[..., None] * (Y * np.log(P + 1e-9) + (1 - Y) * np.log(1 - P + 1e-9))) / norm
            history.append(float(loss))
            dlog = M[..., None] * (P - Y) / norm
            gWx, gWh, gb = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
            gWy, gby = np.zeros_like(self.Wy), np.zeros_like(self.by)
            dh_next = np.zeros((X.shape[1], H))
            dc_next = np.zeros((X.shape[1], H))
            for t in range(len(cache) - 1, -1, -1):
                x, h_prev, c_prev, i, f, o, g, c_new, tc, h_new = cache[t]
                gWy += h_new.T @ dlog[t]
                gby += dlog[t].sum(0)
                dh = dlog[t] @ self.Wy.T + dh_next
                do = dh * tc
                dc = dh * o * (1 - tc * tc) + dc_next
                di, df, dg = dc * g, dc * c_prev, dc * i
                dz = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                     do * o * (1 - o), dg * (1 - g * g)], axis=1)
                gWx += x.T @ dz
                gWh += h_prev.T @ dz
                gb += dz.sum(0)
                dh_next = dz @ self.Wh.T
                dc_next = dc * f
            grads = [gWx, gWh, gb, gWy, gby]
            total = np.sqrt(sum((g * g).sum() for g in grads))
            if total > clip:
                grads = [g * clip / total for g in grads]
            opt.step(grads)
            if verbose and step % 200 == 0:
                print(f"  LSTM step {step:5d}  loss {np.mean(history[-50:]):.4f}")
        return history

    def state_dict(self) -> dict:
        return {"Wx": self.Wx, "Wh": self.Wh, "b": self.b, "Wy": self.Wy, "by": self.by,
                "shape": np.array([self.n_in, self.n_hidden, self.n_out])}

    @classmethod
    def from_state_dict(cls, d) -> "LSTM":
        n_in, n_hidden, n_out = (int(v) for v in d["shape"])
        net = cls(n_in, n_hidden, n_out)
        net.Wx, net.Wh, net.b, net.Wy, net.by = (np.array(d[k]) for k in ("Wx", "Wh", "b", "Wy", "by"))
        return net
