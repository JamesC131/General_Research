"""
Non-Action-Centered Subsystem (NACS): LSTM working memory.

Gomez & Sun (2024): "a single-layer LSTM network in the NACS, with 64 LSTM
units, connected to a standard dense output layer of four nodes, constituted
memory for number sets. The network was trained on many random number
sequences ... Noise was included via a 'temperature' parameter of the LSTM."

Here the four output nodes code a digit 0-9 in 4-bit binary. Each time step's
input is [4 bits, store flag, recall flag, distractor flag]. A trial presents
the items (optionally with distractor steps between them, as in span tasks)
and then asks for serial recall.

Control from the ACS top level: an item is encoded with full strength only
when a top-level memory-updating rule controls encoding. Without that control
(implicit processing), the item reaches the LSTM attenuated (gain < 1) and
with noise. Recall samples each output bit with p = sigmoid(logit / T).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .nets import LSTM

N_BITS = 4
N_IN = N_BITS + 3
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_WEIGHTS = DATA_DIR / "wm_lstm.npz"


def digit_bits(d: int) -> np.ndarray:
    return np.array([(d >> k) & 1 for k in range(N_BITS)], dtype=float)


def bits_digit(bits) -> int:
    return int(sum(int(b) << k for k, b in enumerate(bits)))


def build_trial(digits, gains=None, distractors_between: int = 0, rng=None, noise=0.0):
    """Build one trial as (T, n_in) inputs, (T, 4) targets, (T,) mask."""
    n = len(digits)
    gains = np.ones(n) if gains is None else np.asarray(gains, float)
    rows, tgt, mask = [], [], []
    for i, d in enumerate(digits):
        x = np.zeros(N_IN)
        x[:N_BITS] = digit_bits(d) * gains[i]
        if rng is not None and noise > 0 and gains[i] < 1.0:
            x[:N_BITS] += rng.normal(0.0, noise, N_BITS)
        x[N_BITS] = gains[i]  # store flag carries encoding strength
        rows.append(x); tgt.append(np.zeros(N_BITS)); mask.append(0.0)
        for _ in range(distractors_between):
            x = np.zeros(N_IN)
            if rng is not None:
                x[:N_BITS] = rng.random(N_BITS) * 0.5
            x[N_BITS + 2] = 1.0
            rows.append(x); tgt.append(np.zeros(N_BITS)); mask.append(0.0)
    for d in digits:
        x = np.zeros(N_IN)
        x[N_BITS + 1] = 1.0
        rows.append(x); tgt.append(digit_bits(d)); mask.append(1.0)
    return np.array(rows), np.array(tgt), np.array(mask)


TRIAL_LENGTH = 32  # training trials are left-padded to this length (8 items x 3 steps + 8 recalls)


def pad_trial(X, M, length=TRIAL_LENGTH):
    """
    Left-pad a trial with blank steps, as in training. Training batches are
    left-padded, so the LSTM always starts from the state it reaches after blank
    input; recall must start from that same state.
    """
    pad = max(0, length - len(X))
    return np.vstack([np.zeros((pad, X.shape[1])), X]), np.concatenate([np.zeros(pad), M])


def training_sampler(batch=64, min_len=1, max_len=8, max_distract=1):
    """Random digit sequences of random length, padded to a common length."""
    def sample(rng):
        trials = []
        for _ in range(batch):
            n = int(rng.integers(min_len, max_len + 1))
            k = int(rng.integers(0, max_distract + 1))
            trials.append(build_trial(rng.integers(0, 10, n), distractors_between=k, rng=rng))
        T = max(TRIAL_LENGTH, max(len(t[0]) for t in trials))
        X = np.zeros((T, batch, N_IN)); Y = np.zeros((T, batch, N_BITS)); M = np.zeros((T, batch))
        for b, (x, y, m) in enumerate(trials):  # left-pad so recall ends at the last step
            X[T - len(x):, b] = x; Y[T - len(x):, b] = y; M[T - len(x):, b] = m
        return X, Y, M
    return sample


def train_working_memory(steps=12000, seed=0, verbose=True, path: Path | None = DEFAULT_WEIGHTS) -> LSTM:
    """Train on random digit sequences (1-8 items, 0-2 distractor steps between items)."""
    net = LSTM(N_IN, 64, N_BITS, seed=seed)
    sampler = training_sampler(max_distract=2)
    net.fit(sampler, steps=steps // 2, lr=5e-3, seed=seed, verbose=verbose)
    net.fit(sampler, steps=steps // 2, lr=1e-3, seed=seed + 1, verbose=verbose)   # fine-tune
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **net.state_dict())
    return net


@dataclass
class WorkingMemory:
    """NACS working memory. `temperature` = noise in recall (unrelated to explicitness)."""
    net: LSTM
    temperature: float = 0.3
    implicit_gain: float = 0.3     # encoding strength without top-level control
    implicit_noise: float = 0.3

    @classmethod
    def load(cls, path: Path = DEFAULT_WEIGHTS, **kw) -> "WorkingMemory":
        if not Path(path).exists():
            print("Training working-memory LSTM (first run only)...")
            net = train_working_memory(path=Path(path))
        else:
            net = LSTM.from_state_dict(np.load(path))
        return cls(net, **kw)

    def recall(self, digits, controlled, rng, distractors_between: int = 1) -> list[int]:
        """
        digits: items to remember; controlled: bool per item (True if the ACS top
        level controlled encoding). Returns recalled digits in serial order.
        """
        gains = np.where(np.asarray(controlled, bool), 1.0, self.implicit_gain)
        X, _, M = build_trial(digits, gains, distractors_between, rng, self.implicit_noise)
        X, M = pad_trial(X, M)
        logits, _ = self.net.forward(X[:, None, :])
        logits = logits[M > 0, 0, :]
        p = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        bits = (rng.random(p.shape) < p).astype(int)
        return [bits_digit(b) for b in bits]

    def score(self, digits, controlled, rng, distractors_between: int = 1) -> int:
        """Number of items recalled in the correct serial position."""
        out = self.recall(digits, controlled, rng, distractors_between)
        return int(sum(a == b for a, b in zip(out, digits)))
