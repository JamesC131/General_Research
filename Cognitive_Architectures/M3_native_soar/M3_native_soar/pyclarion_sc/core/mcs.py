"""
Metacognitive Subsystem (MCS): goal setting and control of explicitness.

Goal strength (Gomez & Sun 2024, Eq. 7):
    gs_g = sum_d relevance_{d->g} * ds_d
Goal selection: Boltzmann over goal strengths.

Value of a goal (2024 Eq. 3; 2025 Eq. 4):
    value(g) = sum_d ds_d * satisfaction_d(g)

Utility of explicitness level j (2024 Eq. 2; 2025 Eq. 3):
    U_j = explicitnesslevel_j * (alpha * value(g) - c)
Level selection: Boltzmann over U_j (2024 Eq. 4).

Explicitness levels are coded as integers 0..n_levels-1, "so that a higher
number represented higher explicitness" (Gomez & Sun 2024). The chosen code is
turned into the probability that the ACS top level determines the action:
    p_explicit = level / (n_levels - 1)
Level 0 = fully implicit, top level = fully explicit.

Explicitness may instead be set by the situation (e.g., cognitive load in
Friese et al., 2008), via `fixed_explicitness`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .choice import boltzmann_choice
from .ms import MotivationalSubsystem


@dataclass
class Goal:
    name: str
    relevance: dict[str, float] = field(default_factory=dict)     # drive -> relevance (Eq. 7)
    satisfaction: dict[str, float] = field(default_factory=dict)  # drive -> satisfaction (Eq. 3/4)


@dataclass
class ExplicitnessDecision:
    level: int                 # chosen code (0..n_levels-1)
    p_explicit: float          # probability the top level acts
    value: float               # value(g)
    utilities: np.ndarray      # U_j for every level
    probabilities: np.ndarray  # Boltzmann probabilities of every level
    source: str = "utility"    # "utility" or "situation"


@dataclass
class GoalDecision:
    goal: str
    strengths: dict[str, float]
    probabilities: dict[str, float]


@dataclass
class MetacognitiveSubsystem:
    goals: dict[str, Goal] = field(default_factory=dict)
    alpha: float = 1.0            # benefit coefficient
    cost: float = 0.16            # cost coefficient c
    temperature: float = 0.8      # Boltzmann temperature for explicitness
    goal_temperature: float = 0.1
    n_levels: int = 11            # codes 0..10

    def add_goal(self, name: str, relevance: dict[str, float] | None = None,
                 satisfaction: dict[str, float] | None = None) -> Goal:
        self.goals[name] = Goal(name, dict(relevance or {}), dict(satisfaction or {}))
        return self.goals[name]

    # -- goals --------------------------------------------------------------
    def goal_strengths(self, ms: MotivationalSubsystem) -> dict[str, float]:
        """Eq. 7."""
        ds = ms.strengths()
        return {g.name: sum(r * ds.get(d, 0.0) for d, r in g.relevance.items())
                for g in self.goals.values()}

    def select_goal(self, ms: MotivationalSubsystem, rng: np.random.Generator,
                    candidates: list[str] | None = None) -> GoalDecision:
        gs = self.goal_strengths(ms)
        names = candidates or list(gs)
        idx, p = boltzmann_choice([gs[n] for n in names], self.goal_temperature, rng)
        return GoalDecision(names[idx], gs, dict(zip(names, p.tolist())))

    # -- explicitness ---------------------------------------------------------
    def value(self, goal: str, ms: MotivationalSubsystem) -> float:
        """Eq. 3 (2024) / Eq. 4 (2025)."""
        ds = ms.strengths()
        return sum(ds.get(d, 0.0) * s for d, s in self.goals[goal].satisfaction.items())

    @property
    def levels(self) -> np.ndarray:
        return np.arange(self.n_levels, dtype=float)

    def explicitness_utilities(self, value: float) -> np.ndarray:
        """Eq. 2 (2024) / Eq. 3 (2025)."""
        return self.levels * (self.alpha * value - self.cost)

    def select_explicitness(self, goal: str, ms: MotivationalSubsystem,
                            rng: np.random.Generator) -> ExplicitnessDecision:
        v = self.value(goal, ms)
        U = self.explicitness_utilities(v)
        idx, p = boltzmann_choice(U, self.temperature, rng)
        return ExplicitnessDecision(idx, idx / (self.n_levels - 1), v, U, p)

    def expected_explicitness(self, goal: str, ms: MotivationalSubsystem) -> float:
        """Average p_explicit under the Boltzmann distribution (no sampling)."""
        from .choice import boltzmann_probs
        p = boltzmann_probs(self.explicitness_utilities(self.value(goal, ms)), self.temperature)
        return float((p * self.levels).sum() / (self.n_levels - 1))

    @staticmethod
    def fixed_explicitness(p_explicit: float) -> ExplicitnessDecision:
        """Explicitness dictated by the situation (e.g., cognitive capacity)."""
        return ExplicitnessDecision(-1, float(p_explicit), float("nan"),
                                    np.array([]), np.array([]), source="situation")
