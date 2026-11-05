"""
Action-Centered Subsystem (ACS).

Top level: explicit "state, goal -> action" rules. Applicable rules compete
through a Boltzmann distribution over their support values (default 1).

Bottom level: an implicit network computes Q-values for each action from the
state (and goal); an action is sampled from a Boltzmann distribution over the
Q-values (Gomez & Sun 2025, Eq. 1).

Integration: the MCS's explicitness decision gives p_explicit, the probability
that the top level's action is used. If no rule applies, the bottom level acts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .choice import boltzmann_choice

State = dict[str, Any]


@dataclass
class Rule:
    """
    IF condition(state, goal) THEN action.

    `condition` is either a callable (state, goal) -> bool or a dict of
    required state values; the optional `goal` field must match the goal.
    """
    name: str
    action: Any
    condition: Callable[[State, str | None], bool] | dict = field(default_factory=dict)
    goal: str | None = None
    support: float = 1.0
    description: str = ""

    def applies(self, state: State, goal: str | None) -> bool:
        if self.goal is not None and self.goal != goal:
            return False
        if callable(self.condition):
            return bool(self.condition(state, goal))
        return all(state.get(k) == v for k, v in self.condition.items())

    def __str__(self) -> str:
        return self.description or f"{self.name}: -> {self.action}"


@dataclass
class TopLevel:
    rules: list[Rule] = field(default_factory=list)
    temperature: float = 0.1

    def add(self, *rules: Rule) -> None:
        self.rules.extend(rules)

    def applicable(self, state: State, goal: str | None) -> list[Rule]:
        return [r for r in self.rules if r.applies(state, goal)]

    def decide(self, state: State, goal: str | None, rng) -> tuple[Rule | None, dict]:
        rules = self.applicable(state, goal)
        if not rules:
            return None, {"applicable": []}
        idx, p = boltzmann_choice([r.support for r in rules], self.temperature, rng)
        return rules[idx], {"applicable": [r.name for r in rules], "probabilities": p.tolist()}


@dataclass
class BottomLevel:
    """
    q_function(state, goal) -> array of Q-values, one per action in `actions`.
    Usually wraps an MLP (see make_mlp_q).
    """
    actions: list[Any]
    q_function: Callable[[State, str | None], np.ndarray]
    temperature: float = 0.1

    def decide(self, state: State, goal: str | None, rng) -> tuple[Any, dict]:
        q = np.asarray(self.q_function(state, goal), dtype=float)
        idx, p = boltzmann_choice(q, self.temperature, rng)
        return self.actions[idx], {"q": q.tolist(), "probabilities": p.tolist()}


def make_mlp_q(net, encode: Callable[[State, str | None], np.ndarray]):
    """Wrap an MLP as a Q-function: Q = net(encode(state, goal))."""
    return lambda state, goal: net.predict(encode(state, goal))[0]


@dataclass
class ACSDecision:
    action: Any
    level: str                 # "top" or "bottom"
    rule: str | None
    p_explicit: float
    details: dict


@dataclass
class ActionCenteredSubsystem:
    top: TopLevel
    bottom: BottomLevel

    def decide(self, state: State, goal: str | None, p_explicit: float, rng) -> ACSDecision:
        use_top = rng.random() < p_explicit
        if use_top:
            rule, info = self.top.decide(state, goal, rng)
            if rule is not None:
                return ACSDecision(rule.action, "top", rule.name, p_explicit, info)
        action, info = self.bottom.decide(state, goal, rng)
        return ACSDecision(action, "bottom", None, p_explicit, info)
