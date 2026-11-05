"""
Index data for Robot World: scenarios, conditions, robots, and a comparison of
the three robots averaged over many seeds (descriptive only, no tests).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .controller import CONTROLLERS, SCENARIOS, run_episode

CONTROLLER_LABELS = {"clarion": "Clarion self-control", "plan_only": "Plan only", "habit_only": "Habit only"}
METRICS = [("steps", "Steps to finish"), ("battery_used", "Battery used"), ("planning_calls", "Planning calls"),
           ("unassigned_cleaned_pct", "Room C cleaned (%)"), ("quiet_violations", "Quiet-zone rule broken"),
           ("finished", "Episodes finished")]


@lru_cache(maxsize=None)
def comparison(scenario: str, condition: str, seeds: int = 30) -> dict:
    out = {}
    for ctl in CONTROLLERS:
        rows = [run_episode(scenario, condition, seed=s, controller=ctl)["summary"] for s in range(1, seeds + 1)]
        out[ctl] = {k: float(np.mean([float(r[k]) for r in rows])) for k, _ in METRICS}
    out["seeds"] = seeds
    return out


def index(seeds_available: list[int] | None = None, comparison_seeds: int = 30) -> dict:
    return {
        "scenarios": [{"id": k, "title": v["title"], "paper": v["paper"], "story": v["story"],
                       "conditions": [{"id": c, "label": lab} for c, (lab, _) in v["conditions"].items()]}
                      for k, v in SCENARIOS.items()],
        "controllers": [{"id": k, "label": CONTROLLER_LABELS[k], "description": d} for k, d in CONTROLLERS.items()],
        "metrics": [{"id": k, "label": lab} for k, lab in METRICS],
        "comparison": {sc: {c: comparison(sc, c, comparison_seeds) for c in spec["conditions"]}
                       for sc, spec in SCENARIOS.items()},
        "seeds": seeds_available,
    }
