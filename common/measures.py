"""
Shared by every package: what is measured in an episode, and how each decision is logged.

These functions are copied unchanged from the study code (soar-clarion/experiments/harness.py),
so every package computes its measures exactly as the published analysis did.

    logged(RobotClass)      -> the same robot class, but every decision also records what the robot saw
    episode_measures(ep)    -> one row of measures for one episode (Document 1, Section 6)
    all_conditions()        -> the six (scenario, condition) pairs of the task
"""
from __future__ import annotations

import numpy as np

from pyclarion_sc.robot.controller import SCENARIOS

TEMPTATION_PHASES = ("approaching room C's door", "past room C's door")


def logged(base):
    """Wrap a robot class so that each decision also stores the temptation, room, busy flag and battery."""
    class Logged(base):
        def decide(self, p, phase):
            d = super().decide(p, phase)
            d["temptation"] = float(p["temptation"])
            d["room"] = str(p["room"])
            d["busy"] = bool(self.busy(p))
            d["battery"] = float(p["battery"])
            return d
    Logged.__name__ = "Logged" + base.__name__
    return Logged


def _rate(decs):
    return float(np.mean([d["option"] == "detour" for d in decs])) if decs else float("nan")


def episode_measures(ep) -> dict:
    """Document 1, Section 6. NaN = no exposure of that kind in the episode."""
    real = [d for d in ep["decisions"] if not d.get("reflex")]
    exposed = [d for d in real if d.get("temptation", 0) > 0 and d.get("room") != "C"]
    by = lambda ph: [d for d in exposed if d["phase"] == ph]
    after_a = [d for d in exposed if d["phase"] != "cleaning room A"]
    busy = [d for d in exposed if d.get("busy")]
    s = ep["summary"]

    def mean_p(phases):
        v = [d["p_explicit"] for d in real if d["phase"] in phases]
        return float(np.mean(v)) if v else float("nan")

    def top(decs):
        return float(np.mean([d["level"] == "top" for d in decs])) if decs else float("nan")

    return {
        # primary
        "lapse_rate": _rate(exposed),
        "lapse_approach": _rate(by(TEMPTATION_PHASES[0])),
        "lapse_past": _rate(by(TEMPTATION_PHASES[1])),
        "lapse_after_A": _rate(after_a),
        "lapse_busy": _rate(busy),
        # secondary
        "room_c_cleaned_pct": float(s["unassigned_cleaned_pct"]),
        "steps": float(s["steps"]),
        "finished": float(s["finished"]),
        "planning_calls": float(s["planning_calls"]),
        "battery_used": float(s["battery_used"]),
        "quiet_violations": float(s["quiet_violations"]),
        "top_share": float(s["top_share"]),
        "top_share_exposed": top(exposed),
        # manipulation checks
        "p_explicit_approach": mean_p((TEMPTATION_PHASES[0],)),
        "p_explicit_past": mean_p((TEMPTATION_PHASES[1],)),
        "p_explicit_after_A": mean_p(("approaching room C's door", "past room C's door", "cleaning room B")),
        "deficit_entering_B": next((d["drives"]["achievement"]["deficit"] for d in real
                                    if d["phase"] == "cleaning room B" and d.get("drives")), float("nan")),
        "n_exposed": len(exposed),
        "n_deliberations": sum(bool(d.get("deliberated")) for d in real),
        "n_timeouts": sum(bool(d.get("timed_out")) for d in real),
    }


def all_conditions():
    return [(sc, c) for sc, spec in SCENARIOS.items() for c in spec["conditions"]]
