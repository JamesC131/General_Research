"""
A Clarion agent: MS + MCS + ACS (+ optional NACS), with one decision cycle.

The agent never talks to a specific world. It receives a *perception* and
returns an *action*, so the same agent can run a lab experiment, a 3D
simulation or a real robot:

    perception -> MS drive strengths -> MCS goal + explicitness
               -> ACS top/bottom action -> world

Perception format (a plain dict):
    {
      "state":    {...},                      # features the rules/networks read
      "stimulus": {"food": 0.8, ...},          # optional: drive stimulus levels
      "goal":     "choose_snack",              # optional: goal set by the world
      "explicitness": 0.85,                    # optional: set by the situation
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .core.acs import ActionCenteredSubsystem
from .core.mcs import MetacognitiveSubsystem
from .core.ms import MotivationalSubsystem
from .core.nacs import WorkingMemory


class Environment(Protocol):
    """Anything the agent can act in: an experiment, a simulator, a robot."""

    def reset(self) -> dict: ...
    def step(self, action: Any) -> tuple[dict, float, bool]: ...   # perception, reward, done


@dataclass
class StepRecord:
    t: int
    goal: str | None
    drives: dict
    value: float
    explicitness_level: int
    p_explicit: float
    level_used: str
    rule: str | None
    action: Any

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["drives"] = {k: v["strength"] for k, v in self.drives.items()}
        return d


@dataclass
class ClarionAgent:
    ms: MotivationalSubsystem
    mcs: MetacognitiveSubsystem
    acs: ActionCenteredSubsystem
    nacs: WorkingMemory | None = None
    name: str = "agent"
    seed: int | None = None
    trace: list[StepRecord] = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._t = 0

    def decide(self, perception: dict) -> StepRecord:
        """One full Clarion cycle."""
        if "stimulus" in perception:
            self.ms.set_stimulus(**perception["stimulus"])
        goal = perception.get("goal")
        if goal is None and self.mcs.goals:
            goal = self.mcs.select_goal(self.ms, self.rng).goal
        if perception.get("explicitness") is not None:
            ex = self.mcs.fixed_explicitness(perception["explicitness"])
        else:
            ex = self.mcs.select_explicitness(goal, self.ms, self.rng)
        dec = self.acs.decide(perception.get("state", {}), goal, ex.p_explicit, self.rng)
        rec = StepRecord(self._t, goal, self.ms.snapshot(), ex.value, ex.level,
                         ex.p_explicit, dec.level, dec.rule, dec.action)
        self.trace.append(rec)
        self._t += 1
        return rec

    def run(self, env: Environment, max_steps: int = 1000) -> float:
        """Run one episode. Returns total reward."""
        perception, total = env.reset(), 0.0
        for _ in range(max_steps):
            rec = self.decide(perception)
            perception, reward, done = env.step(rec.action)
            total += reward
            if done:
                break
        return total
