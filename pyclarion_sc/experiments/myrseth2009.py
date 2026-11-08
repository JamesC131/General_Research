"""
Simulations 3 and 4: Myrseth, Fishbach & Trope (2009), counteractive self-control.
Source: Gomez & Sun (2025), "Self-Control in Context: A Cognitive Architectural Approach".

34 participants rate the appeal (0-150) of a plate of 3 granola bars and a
plate of 3 chocolate bars, either before choosing a snack (n = 16) or after
(n = 18). Human result: granola > chocolate before choice; no difference after.

Interpretation 1 ("explicit_implicit"): the achievement drive is more
stimulated before choice (0.8 vs 0.5), so value(g) and the utility of high
explicitness are higher, the top-level (healthy) rules act more often, and
chocolate is devalued.

Interpretation 2 ("desire_goal"): same explicitness mechanism, plus top-level
subgoal-creation rules ("choose healthy food") that inform both levels; the
bottom network takes the active goal as input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..agent import ClarionAgent
from ..analysis import stats as S
from ..core.acs import ActionCenteredSubsystem, BottomLevel, Rule, TopLevel
from ..core.mcs import MetacognitiveSubsystem
from ..core.ms import MotivationalSubsystem
from ..core.nets import MLP
from .common import N_LEVELS, P

ITEMS = ["granola_1", "granola_2", "granola_3", "chocolate_1", "chocolate_2", "chocolate_3"]
GOALS = ["choose_snack", "choose_healthy_food", "choose_tasty_food"]

HUMAN = {  # Myrseth et al. (2009) as reported in Gomez & Sun (2025)
    "before": {"granola": (102.19, 19.58), "chocolate": (74.06, 32.44), "t": 2.82, "d": 1.05, "n": 16},
    "after": {"granola": (94.22, 26.11), "chocolate": (93.11, 39.80), "t": 0.124, "d": 0.03, "n": 18},
}
PUBLISHED_SIM = {
    "explicit_implicit": {"before": {"granola": (101.5, 20.3), "chocolate": (75.2, 31.8), "t": 2.78, "d": 1.02},
                          "after": {"granola": (93.8, 25.9), "chocolate": (92.9, 39.2), "t": 0.131, "d": 0.03},
                          "F": 4.15},
    "desire_goal": {"before": {"granola": (101.5, 18.9), "chocolate": (75.2, 31.8), "t": 2.75, "d": 1.02},
                    "after": {"granola": (93.8, 25.8), "chocolate": (92.9, 38.2), "t": 0.118, "d": 0.03},
                    "F": 4.12},
}


@dataclass
class MyrsethConfig:
    interpretation: str = P("explicit_implicit", "paper", "'explicit_implicit' (Sec. 4.1) or 'desire_goal' (Sec. 4.2)")
    n_before: int = P(16, "paper")
    n_after: int = P(18, "paper")
    # MS (Eq. 2)
    food_stimulus: float = P(0.8, "paper")
    food_deficit: float = P(0.7, "paper")
    food_satisfaction: float = P(0.7, "paper", "satisfaction of the food drive by either food (constant)")
    achievement_stimulus_before: float = P(0.8, "paper")
    achievement_stimulus_after: float = P(0.5, "paper")
    achievement_deficit: float = P(0.7, "paper")
    achievement_satisfaction: float = P(0.5, "paper", "satisfaction of achievement by the 'choose snack' goal")
    # MCS (Eq. 3)
    alpha: float = P(1.0, "paper")
    cost: float = P(0.185, "paper")
    temperature: float = P(0.7, "paper")
    n_levels: int = P(N_LEVELS, "chosen", "explicitness codes 0..20; not given in the paper")
    # ACS top level: explicit evaluations implied by the healthy-choice rules
    top_granola_mean: float = P(111.1, "calibrated")
    top_granola_sd: float = P(12.2, "calibrated")
    top_chocolate_mean: float = P(53.4, "calibrated")
    top_chocolate_sd: float = P(39.1, "calibrated")
    # ACS bottom level: implicit preferences learned by each participant's network
    bottom_granola_mean: float = P(79.4, "calibrated")
    bottom_granola_sd: float = P(41.8, "calibrated")
    bottom_chocolate_mean: float = P(140.0, "calibrated")
    bottom_chocolate_sd: float = P(50.0, "calibrated")
    healthy_goal_shift: float = P(30.5, "calibrated",
                                  "desire_goal: how much the 'choose healthy food' goal input moves bottom-level ratings toward granola")
    tasty_goal_shift: float = P(10.0, "chosen", "desire_goal: 'choose tasty food' moves ratings toward chocolate")
    timing_effect: float = P(0.0, "chosen", "direct effect of the timing input on bottom ratings (0 = none)")
    response_noise: float = P(6.0, "chosen", "SD of motor/reporting noise on each rating")
    items_per_plate: int = P(3, "paper")
    hidden: tuple = P((8, 8), "chosen", "two ReLU hidden layers (sizes not given in the paper)")
    epochs: int = P(200, "chosen")
    choice_temperature: float = P(0.1, "chosen", "bottom-level Boltzmann temperature for snack choice")
    train_networks: bool = P(True, "chosen",
                             "False = skip training and read the target preferences directly (calibration only)")


def default_config(interpretation: str = "explicit_implicit", **overrides) -> MyrsethConfig:
    """Calibrated defaults for each interpretation (see REPORT.md)."""
    base = dict(CALIBRATED[interpretation])
    base.update(overrides)
    return MyrsethConfig(interpretation=interpretation, **base)


# Written by scripts/calibrate_myrseth.py (fitted to the HUMAN means and SDs).
_CAL_PATH = Path(__file__).resolve().parent.parent / "data" / "myrseth_calibration.json"
CALIBRATED: dict[str, dict] = (json.loads(_CAL_PATH.read_text()) if _CAL_PATH.exists()
                               else {"explicit_implicit": {}, "desire_goal": {}})


# ---------------------------------------------------------------------------
# participant construction
# ---------------------------------------------------------------------------

def _encode(availability, before: bool, goal: str | None, cfg: MyrsethConfig) -> np.ndarray:
    x = list(availability) + [1.0 if before else 0.0]
    if cfg.interpretation == "desire_goal":
        x += [1.0 if goal == g else 0.0 for g in GOALS]
    return np.array(x, float)


def train_bottom_network(prefs: dict, cfg: MyrsethConfig, seed: int) -> MLP:
    """
    Train one participant's implicit-preference network on synthetic experience:
    every combination of available items x timing (x goal), target = this
    participant's implicit ratings (scaled to 0-1).
    """
    rng = np.random.default_rng(seed)
    goals = GOALS if cfg.interpretation == "desire_goal" else [None]
    X, Y = [], []
    for mask in range(64):
        avail = [(mask >> k) & 1 for k in range(6)]
        for before in (True, False):
            for g in goals:
                gr, ch = prefs["granola"], prefs["chocolate"]
                if not before:
                    ch += cfg.timing_effect
                if g == "choose_healthy_food":
                    gr, ch = gr + cfg.healthy_goal_shift, ch - cfg.healthy_goal_shift
                elif g == "choose_tasty_food":
                    gr, ch = gr - cfg.tasty_goal_shift, ch + cfg.tasty_goal_shift
                X.append(_encode(avail, before, g, cfg))
                Y.append([gr / 150, ch / 150])
    X, Y = np.array(X), np.array(Y) + rng.normal(0, 0.01, (len(Y), 2))
    if not cfg.train_networks:
        return _Lookup(X, Y)
    net = MLP((X.shape[1], *cfg.hidden, 2), hidden="relu", output="linear", seed=seed)
    net.fit(X, Y, epochs=cfg.epochs, lr=0.02, seed=seed)  # full-batch Adam
    return net


class _Lookup:
    """Stand-in for a perfectly trained network (calibration speed-up only)."""
    def __init__(self, X, Y):
        self.table = {tuple(x): y for x, y in zip(X, Y)}

    def predict(self, x):
        return np.atleast_2d(self.table[tuple(np.asarray(x, float))])


def build_agent(cfg: MyrsethConfig, before: bool, seed: int) -> tuple[ClarionAgent, dict]:
    rng = np.random.default_rng(seed)
    traits = {
        "top": {"granola": rng.normal(cfg.top_granola_mean, cfg.top_granola_sd),
                "chocolate": rng.normal(cfg.top_chocolate_mean, cfg.top_chocolate_sd)},
        "bottom": {"granola": rng.normal(cfg.bottom_granola_mean, cfg.bottom_granola_sd),
                   "chocolate": rng.normal(cfg.bottom_chocolate_mean, cfg.bottom_chocolate_sd)},
    }
    ms = MotivationalSubsystem()
    ms.add("food", cfg.food_stimulus, cfg.food_deficit)
    ms.add("achievement", cfg.achievement_stimulus_before if before else cfg.achievement_stimulus_after,
           cfg.achievement_deficit)

    mcs = MetacognitiveSubsystem(alpha=cfg.alpha, cost=cfg.cost, temperature=cfg.temperature,
                                 n_levels=cfg.n_levels)
    mcs.add_goal("choose_snack", relevance={"food": 1.0, "achievement": 1.0},
                 satisfaction={"achievement": cfg.achievement_satisfaction})

    net = train_bottom_network(traits["bottom"], cfg, seed + 7919)
    top = TopLevel(_rules(cfg))

    def q(state, goal):  # Q-values for the two snack actions = implicit appeal
        r = net.predict(_encode(state["availability"], before, goal, cfg))[0] * 150
        return r / 150 / cfg.choice_temperature * 0.1

    bottom = BottomLevel(["select_granola", "select_chocolate"], q, temperature=0.1)
    agent = ClarionAgent(ms, mcs, ActionCenteredSubsystem(top, bottom), seed=seed)
    agent.net, agent.traits, agent.before = net, traits, before
    return agent, traits


def _rules(cfg: MyrsethConfig) -> list[Rule]:
    g_av = lambda s: s["granola_available"]
    c_av = lambda s: s["chocolate_available"]
    if cfg.interpretation == "explicit_implicit":
        return [
            Rule("R1", "select_granola", lambda s, g: g_av(s) and not s["all_selected"],
                 description="IF (granola bars available) THEN select granola bar"),
            Rule("R2", "select_chocolate", lambda s, g: not g_av(s) and c_av(s) and not s["all_selected"],
                 description="IF (granola bars not available AND chocolates available) THEN select chocolate"),
            Rule("R3", "finish", lambda s, g: not g_av(s) and not c_av(s),
                 description="IF (granola bars not available AND chocolates not available) THEN finish selection"),
            Rule("R4", "finish", lambda s, g: s["all_selected"],
                 description="IF (all items selected) THEN finish selection"),
        ]
    return [
        Rule("SG1", "set_subgoal:choose_healthy_food",
             lambda s, g: s.get("phase") == "subgoal" and g == "choose_snack" and g_av(s),
             description="IF (goal = choose snack) THEN set subgoal = choose healthy food"),
        Rule("SG2", "set_subgoal:choose_tasty_food",
             lambda s, g: s.get("phase") == "subgoal" and g == "choose_snack" and not g_av(s),
             description="IF (goal = choose snack AND healthy food not available) THEN set subgoal = choose tasty food"),
        Rule("S1", "select_granola", lambda s, g: g == "choose_healthy_food" and g_av(s) and not s["all_selected"],
             description="IF (subgoal = choose healthy food AND granola bars available) THEN select a granola bar"),
        Rule("S2", "select_chocolate", lambda s, g: g == "choose_healthy_food" and not g_av(s) and not s["all_selected"],
             description="IF (subgoal = choose healthy food AND no granola bars available) THEN select chocolate"),
        Rule("S3", "select_chocolate", lambda s, g: g == "choose_tasty_food" and c_av(s) and not s["all_selected"],
             description="IF (subgoal = choose tasty food AND chocolates available) THEN select chocolate"),
        Rule("S4", "select_granola", lambda s, g: g == "choose_tasty_food" and not c_av(s) and not s["all_selected"],
             description="IF (subgoal = choose tasty food AND no chocolates available) THEN select granola bar"),
        Rule("S5", "finish", lambda s, g: s["all_selected"],
             description="IF (all items selected) THEN finish selection"),
    ]


# ---------------------------------------------------------------------------
# one participant
# ---------------------------------------------------------------------------

def _state(phase="evaluate", all_selected=False):
    return {"phase": phase, "availability": [1.0] * 6, "granola_available": True,
            "chocolate_available": True, "all_selected": all_selected}


def run_participant(cfg: MyrsethConfig, before: bool, seed: int) -> dict:
    agent, traits = build_agent(cfg, before, seed)
    rng = agent.rng
    events = []
    goal = "choose_snack"

    # Desire-goal: the top level may create the "choose healthy food" subgoal.
    if cfg.interpretation == "desire_goal":
        ex = agent.mcs.select_explicitness("choose_snack", agent.ms, rng)
        dec = agent.acs.decide(_state("subgoal"), "choose_snack", ex.p_explicit, rng)
        if dec.level == "top" and str(dec.action).startswith("set_subgoal:"):
            goal = dec.action.split(":", 1)[1]
        events.append({"step": "subgoal", "explicitness": ex.p_explicit, "level": dec.level, "goal": goal})

    def choose():
        ex = agent.mcs.select_explicitness("choose_snack", agent.ms, rng)
        dec = agent.acs.decide(_state("choose"), goal, ex.p_explicit, rng)
        events.append({"step": "choice", "explicitness": ex.p_explicit, "level": dec.level,
                       "rule": dec.rule, "action": dec.action})
        return dec.action

    choice = None
    if not before:
        choice = choose()

    ratings = {}
    for food in ("granola", "chocolate"):
        values = []
        for k in range(cfg.items_per_plate):
            ex = agent.mcs.select_explicitness("choose_snack", agent.ms, rng)
            use_top = rng.random() < ex.p_explicit
            rule = None
            if use_top:
                # Evaluating a plate = "what would my rules do here?" (both foods on the table)
                rule, _ = agent.acs.top.decide(_state(), goal, rng)
            if use_top and rule is not None and rule.action in ("select_granola", "select_chocolate"):
                # Explicit evaluation: an applicable healthy-choice rule means the plate is
                # judged by the participant's explicit (health-based) attitude.
                v = traits["top"][food]
                level = "top"
            else:
                # Implicit evaluation: the participant's trained network.
                out = agent.net.predict(_encode([1.0] * 6, before, goal, cfg))[0] * 150
                v = out[0] if food == "granola" else out[1]
                level = "bottom"
            v = float(np.clip(v + rng.normal(0, cfg.response_noise), 0, 150))
            values.append(v)
            events.append({"step": f"rate_{food}_{k + 1}", "explicitness": ex.p_explicit, "level": level,
                           "rule": rule.name if (level == "top" and rule) else None, "rating": v})
        ratings[food] = float(np.mean(values))

    if before:
        choice = choose()

    top_share = np.mean([e["level"] == "top" for e in events if e["step"].startswith("rate")])
    return {"condition": "before" if before else "after", "granola": ratings["granola"],
            "chocolate": ratings["chocolate"], "choice": choice, "goal": goal,
            "mean_explicitness": float(np.mean([e["explicitness"] for e in events])),
            "top_level_share": float(top_share),
            "achievement_strength": agent.ms["achievement"].strength,
            "value": agent.mcs.value("choose_snack", agent.ms),
            "events": events}


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def run(cfg: MyrsethConfig | None = None, seed: int = 1) -> dict:
    cfg = cfg or default_config()
    parts = []
    for i in range(cfg.n_before):
        parts.append(run_participant(cfg, True, seed * 100003 + i))
    for i in range(cfg.n_after):
        parts.append(run_participant(cfg, False, seed * 100003 + 5000 + i))
    return {"config": cfg, "participants": parts, "stats": analyze(parts)}


def analyze(parts: list[dict]) -> dict:
    out = {}
    for cond in ("before", "after"):
        g = [p["granola"] for p in parts if p["condition"] == cond]
        c = [p["chocolate"] for p in parts if p["condition"] == cond]
        out[cond] = {"granola": S.describe(g), "chocolate": S.describe(c), "test": S.paired_t(g, c),
                     "chose_granola": float(np.mean([p["choice"] == "select_granola"
                                                      for p in parts if p["condition"] == cond])),
                     "mean_explicitness": float(np.mean([p["mean_explicitness"] for p in parts
                                                         if p["condition"] == cond]))}
    b = [p for p in parts if p["condition"] == "before"]
    a = [p for p in parts if p["condition"] == "after"]
    out["interaction"] = S.mixed_anova_interaction([p["granola"] for p in b], [p["chocolate"] for p in b],
                                                   [p["granola"] for p in a], [p["chocolate"] for p in a])
    return out
