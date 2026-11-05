"""
M3, the native Soar model, and its variants.

The model is native_cleaner.soar (45 productions) with the trained habit habit_rl.soar (41 Soar-RL rules),
run through the bridge native_cleaner.py in the same grid world and episode loop as CLARION.

  native_soar               the model: Soar-RL habit; a conflict triggers deliberation in a substate; chunking on
  native_habit_only         baseline: never deliberate (the Soar-RL habit always chooses)
  native_always_deliberate  baseline: deliberate on every cycle, conflict or not
  N1_no_chunking            ablation: chunking off, so every conflict is deliberated in full

CLARION's ablations A1 (fixed drives) and A2 (no cost) do not apply: M3 has no drives and no cost term.
"""
from __future__ import annotations

from common.measures import episode_measures, logged
from pyclarion_sc.robot.controller import make_config

from .native_cleaner import NativeSoarCleaner

VARIANTS = {
    "native_soar": {},
    "native_habit_only": {"mode": "habit"},
    "native_always_deliberate": {"mode": "always"},
    "N1_no_chunking": {"chunking": False},
}


def build(variant, scenario, condition, seed, overrides=None):
    spec = VARIANTS[variant]
    cfg = make_config(scenario, condition, **(overrides or {}))
    return logged(NativeSoarCleaner)(cfg, seed=seed, **spec)


def run_one(variant, scenario, condition, seed, overrides=None, return_episode=False):
    robot = build(variant, scenario, condition, seed, overrides)
    try:
        ep = robot.run()
    finally:
        robot.close()
    m = episode_measures(ep)
    m.update({"variant": variant, "scenario": scenario, "condition": condition, "seed": seed})
    return (m, ep) if return_episode else m


def trace_row(d):
    if d.get("reflex"):
        return {"t": d["t"], "phase": d["phase"], "chosen": "dock", "level": "reflex"}
    return {"t": d["t"], "phase": d["phase"], "room": d.get("room"), "battery": round(d.get("battery", 0), 3),
            "temptation": round(d.get("temptation", 0.0), 3), "conflict": d["conflict"],
            "deliberated": d["deliberated"], "decision_cycles_used": d["decisions_used"],
            "budget": d["budget"] if d["budget"] < 60 else "none", "timed_out": d["timed_out"],
            "chosen": d["option"], "chosen_by": "control knowledge" if d["level"] == "top" else "habit (Soar-RL)"}
