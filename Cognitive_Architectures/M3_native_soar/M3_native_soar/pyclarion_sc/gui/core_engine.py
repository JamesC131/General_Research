"""
Step-by-step Clarion cycle for Core Studio.

Runs the same equations as pyclarion_sc.core, but records every intermediate
value (drive strengths, goal strengths and probabilities, value(g), utility
of each explicitness level, level probabilities, the top level's and the
bottom level's candidate actions, and which level acted).

A scenario is plain JSON so it can be edited in the browser:

{
  "name": "...", "description": "...",
  "drives":  [{"name": "achievement", "stimulus": 0.8, "deficit": 0.7, "k": 0.0}],
  "goals":   [{"name": "choose_snack", "relevance": {"achievement": 1}, "satisfaction": {"achievement": 0.5}}],
  "mcs":     {"alpha": 1, "cost": 0.185, "temperature": 0.7, "goal_temperature": 0.1,
              "n_levels": 21, "mode": "utility", "fixed": 0.8, "dt": 0.05},
  "state":   {"granola_available": true},
  "actions": ["select_granola", "select_chocolate"],
  "q_values": {"select_granola": 0.2, "select_chocolate": 0.8},
  "bottom_temperature": 0.1,
  "rules":   [{"name": "R1", "if": {"granola_available": true}, "goal": null,
               "then": "select_granola", "support": 1.0}],
  "top_temperature": 0.1
}

Each drive's "k" is its deficit reduction rate (Eq. 6) per unit of time on
task; "dt" is the time that passes per decision cycle.
"""
from __future__ import annotations

import copy

import numpy as np

from ..core.choice import boltzmann_probs
from ..core.mcs import MetacognitiveSubsystem
from ..core.ms import MotivationalSubsystem

LIMITS = {"steps": 500, "drives": 12, "goals": 12, "rules": 40, "actions": 20, "n_levels": 101}


def _f(x, lo=-1e6, hi=1e6) -> float:
    return float(min(max(float(x), lo), hi))


def validate(sc: dict) -> dict:
    """Clean a scenario coming from the browser; raises ValueError with a readable message."""
    sc = copy.deepcopy(sc)
    if not sc.get("drives"):
        raise ValueError("Add at least one drive.")
    if not sc.get("goals"):
        raise ValueError("Add at least one goal.")
    if not sc.get("actions"):
        raise ValueError("Add at least one action.")
    for key in ("drives", "goals", "rules", "actions"):
        if len(sc.get(key, [])) > LIMITS[key]:
            raise ValueError(f"Too many {key} (limit {LIMITS[key]}).")
    names = [d["name"] for d in sc["drives"]]
    if len(set(names)) != len(names):
        raise ValueError("Drive names must be unique.")
    actions = list(sc["actions"])
    for r in sc.get("rules", []):
        if r.get("then") not in actions:
            raise ValueError(f"Rule '{r.get('name')}' points to an action that does not exist: {r.get('then')}.")
    m = sc.setdefault("mcs", {})
    m["n_levels"] = int(min(max(int(m.get("n_levels", 21)), 2), LIMITS["n_levels"]))
    for k, lo in (("temperature", 1e-3), ("goal_temperature", 1e-3)):
        m[k] = _f(m.get(k, 0.7 if k == "temperature" else 0.1), lo, 100)
    sc["bottom_temperature"] = _f(sc.get("bottom_temperature", 0.1), 1e-3, 100)
    sc["top_temperature"] = _f(sc.get("top_temperature", 0.1), 1e-3, 100)
    return sc


def run(scenario: dict, steps: int = 40, seed: int = 1) -> dict:
    sc = validate(scenario)
    steps = int(min(max(steps, 1), LIMITS["steps"]))
    rng = np.random.default_rng(int(seed))
    m = sc["mcs"]

    ms = MotivationalSubsystem()
    for d in sc["drives"]:
        ms.add(d["name"], _f(d.get("stimulus", 0)), _f(d.get("deficit", 0), 0, 10))
    mcs = MetacognitiveSubsystem(alpha=_f(m.get("alpha", 1)), cost=_f(m.get("cost", 0.16)),
                                 temperature=m["temperature"], goal_temperature=m["goal_temperature"],
                                 n_levels=m["n_levels"])
    for g in sc["goals"]:
        mcs.add_goal(g["name"], g.get("relevance", {}), g.get("satisfaction", {}))

    actions = list(sc["actions"])
    q = np.array([_f(sc.get("q_values", {}).get(a, 0.0)) for a in actions])
    state = dict(sc.get("state", {}))
    dt = _f(m.get("dt", 0.05), 0, 10)
    out = []

    for t in range(steps):
        # Eq. 6: deficits fall with time on task
        for d in sc["drives"]:
            k = _f(d.get("k", 0.0), 0, 10)
            if k > 0:
                ms.reduce_deficit(d["name"], k, (t + 1) * dt)
        drives = ms.snapshot()

        # Eq. 7: goal strengths -> Boltzmann goal choice
        gs = mcs.goal_strengths(ms)
        gnames = list(gs)
        gp = boltzmann_probs([gs[n] for n in gnames], mcs.goal_temperature)
        goal = gnames[int(rng.choice(len(gnames), p=gp))]

        # Eqs. 2-4: value, utilities, explicitness choice
        value = mcs.value(goal, ms)
        U = mcs.explicitness_utilities(value)
        if m.get("mode", "utility") == "fixed":
            p_levels = None
            level = -1
            p_explicit = _f(m.get("fixed", 0.8), 0, 1)
        else:
            p_levels = boltzmann_probs(U, mcs.temperature)
            level = int(rng.choice(len(U), p=p_levels))
            p_explicit = level / (mcs.n_levels - 1)

        # ACS top level: applicable rules -> Boltzmann over supports
        applicable = [r for r in sc.get("rules", [])
                      if (r.get("goal") in (None, "", goal))
                      and all(bool(state.get(k)) == bool(v) for k, v in r.get("if", {}).items())]
        top_choice, top_rule, top_probs = None, None, []
        if applicable:
            tp = boltzmann_probs([_f(r.get("support", 1.0)) for r in applicable], sc["top_temperature"])
            i = int(rng.choice(len(applicable), p=tp))
            top_rule, top_choice = applicable[i]["name"], applicable[i]["then"]
            top_probs = tp.tolist()

        # ACS bottom level: Q-values -> Boltzmann
        bp = boltzmann_probs(q, sc["bottom_temperature"])
        bottom_choice = actions[int(rng.choice(len(actions), p=bp))]

        # integration: top acts with probability p_explicit (if a rule applies)
        use_top = top_choice is not None and rng.random() < p_explicit
        action = top_choice if use_top else bottom_choice
        out.append({
            "t": t + 1,
            "drives": drives,
            "goal_strengths": gs, "goal_probs": dict(zip(gnames, gp.tolist())), "goal": goal,
            "value": value, "utilities": U.tolist(),
            "level_probs": None if p_levels is None else p_levels.tolist(),
            "level": level, "p_explicit": p_explicit,
            "expected_explicitness": None if p_levels is None else float((p_levels * mcs.levels).sum() / (mcs.n_levels - 1)),
            "applicable_rules": [r["name"] for r in applicable], "top_rule_probs": top_probs,
            "top_rule": top_rule, "top_choice": top_choice,
            "q_values": dict(zip(actions, q.tolist())), "bottom_probs": dict(zip(actions, bp.tolist())),
            "bottom_choice": bottom_choice,
            "level_used": "top" if use_top else "bottom", "action": action,
            "conflict": top_choice is not None and top_choice != bottom_choice,
        })

    summary = {
        "top_share": float(np.mean([s["level_used"] == "top" for s in out])),
        "conflicts": int(sum(s["conflict"] for s in out)),
        "conflicts_won_by_top": int(sum(s["conflict"] and s["level_used"] == "top" for s in out)),
        "actions": {a: int(sum(s["action"] == a for s in out)) for a in actions},
    }
    return {"scenario": sc, "steps": out, "summary": summary}


# ---------------------------------------------------------------------------
# presets (drawn from the papers, plus a robot example)
# ---------------------------------------------------------------------------

def _myrseth(stim: float, label: str) -> dict:
    return {
        "name": f"Myrseth et al. (2009): {label}",
        "description": "Snack choice. The achievement drive is more stimulated when a choice is still ahead "
                       "(0.8) than after it (0.5). Parameters from Gomez & Sun (2025).",
        "drives": [{"name": "food", "stimulus": 0.8, "deficit": 0.7, "k": 0.0},
                   {"name": "achievement", "stimulus": stim, "deficit": 0.7, "k": 0.0}],
        "goals": [{"name": "choose_snack", "relevance": {"food": 1.0, "achievement": 1.0},
                   "satisfaction": {"achievement": 0.5}}],
        "mcs": {"alpha": 1.0, "cost": 0.185, "temperature": 0.7, "goal_temperature": 0.1, "n_levels": 21,
                "mode": "utility", "fixed": 0.8, "dt": 0.05},
        "state": {"granola_available": True, "chocolate_available": True, "all_selected": False},
        "actions": ["select_granola", "select_chocolate", "finish"],
        "q_values": {"select_granola": 0.35, "select_chocolate": 0.6, "finish": 0.0},
        "bottom_temperature": 0.1,
        "rules": [
            {"name": "R1", "if": {"granola_available": True, "all_selected": False}, "goal": None,
             "then": "select_granola", "support": 1.0,
             "text": "IF granola bars available THEN select granola bar"},
            {"name": "R2", "if": {"granola_available": False, "chocolate_available": True}, "goal": None,
             "then": "select_chocolate", "support": 1.0,
             "text": "IF granola not available AND chocolates available THEN select chocolate"},
            {"name": "R3", "if": {"granola_available": False, "chocolate_available": False}, "goal": None,
             "then": "finish", "support": 1.0, "text": "IF nothing available THEN finish selection"},
            {"name": "R4", "if": {"all_selected": True}, "goal": None, "then": "finish", "support": 1.0,
             "text": "IF all items selected THEN finish selection"},
        ],
        "top_temperature": 0.1,
    }


PRESETS = [
    _myrseth(0.8, "before choice"),
    _myrseth(0.5, "after choice"),
    {
        "name": "Schmeichel (2007): attention control over time",
        "description": "Ignoring distractor words satisfies the achievement drive (0.9), so its deficit falls "
                       "with time on task (Eq. 6, k = 0.54) and explicit control fades. Gomez & Sun (2024).",
        "drives": [{"name": "achievement", "stimulus": 0.7, "deficit": 1.0, "k": 0.54}],
        "goals": [{"name": "participate", "relevance": {"achievement": 1.0}, "satisfaction": {"achievement": 0.9}}],
        "mcs": {"alpha": 1.0, "cost": 0.16, "temperature": 0.8, "goal_temperature": 0.1, "n_levels": 21,
                "mode": "utility", "fixed": 0.8, "dt": 0.035},
        "state": {"ignore_words": True, "words_on_screen": True},
        "actions": ["watch_and_do_not_read", "watch_and_read_words", "read_words_only"],
        "q_values": {"watch_and_do_not_read": 0.35, "watch_and_read_words": 0.65, "read_words_only": 0.3},
        "bottom_temperature": 0.1,
        "rules": [{"name": "RI1", "if": {"ignore_words": True, "words_on_screen": True}, "goal": None,
                   "then": "watch_and_do_not_read", "support": 1.0,
                   "text": "IF instruction = ignore words AND words on screen THEN watch and do not read"}],
        "top_temperature": 0.1,
    },
    {
        "name": "Robot: task or charger",
        "description": "A robot sorting objects, tempted by a low battery and a curious new object. "
                       "A starting point for your own robot models (not from the papers).",
        "drives": [{"name": "achievement", "stimulus": 0.8, "deficit": 0.8, "k": 0.2},
                   {"name": "energy", "stimulus": 0.4, "deficit": 0.6, "k": 0.0},
                   {"name": "curiosity", "stimulus": 0.6, "deficit": 0.5, "k": 0.0}],
        "goals": [{"name": "finish_task", "relevance": {"achievement": 1.0}, "satisfaction": {"achievement": 0.7}},
                  {"name": "recharge", "relevance": {"energy": 1.0}, "satisfaction": {"energy": 0.9}}],
        "mcs": {"alpha": 1.0, "cost": 0.2, "temperature": 0.7, "goal_temperature": 0.15, "n_levels": 21,
                "mode": "utility", "fixed": 0.8, "dt": 0.05},
        "state": {"objects_left": True, "battery_low": False, "new_object_seen": True},
        "actions": ["sort_next_object", "drive_to_charger", "inspect_new_object"],
        "q_values": {"sort_next_object": 0.3, "drive_to_charger": 0.2, "inspect_new_object": 0.6},
        "bottom_temperature": 0.15,
        "rules": [
            {"name": "T1", "if": {"objects_left": True}, "goal": "finish_task", "then": "sort_next_object",
             "support": 1.0, "text": "IF objects left AND goal = finish task THEN sort next object"},
            {"name": "C1", "if": {}, "goal": "recharge", "then": "drive_to_charger", "support": 1.0,
             "text": "IF goal = recharge THEN drive to charger"},
        ],
        "top_temperature": 0.1,
    },
]
