"""
Stochastic choice used everywhere in Clarion.

Boltzmann (softmax) distribution, Eq. 4 in Gomez & Sun (2024) and Eq. 1 in
Gomez & Sun (2025):

    p(j) = exp(U_j / tau) / sum_k exp(U_k / tau)
"""
from __future__ import annotations

import numpy as np


def boltzmann_probs(values, temperature: float) -> np.ndarray:
    """Convert values (utilities, Q-values, supports, strengths) to probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    v = np.asarray(values, dtype=float) / temperature
    v = v - v.max()  # numerical stability; does not change the result
    e = np.exp(v)
    return e / e.sum()


def boltzmann_choice(values, temperature: float, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Sample one index from the Boltzmann distribution. Returns (index, probabilities)."""
    p = boltzmann_probs(values, temperature)
    return int(rng.choice(len(p), p=p)), p
