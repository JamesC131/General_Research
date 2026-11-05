"""
A cleaning robot controlled by the Clarion self-control architecture.

Every few steps the robot runs one Clarion cycle:

  perception -> MS: drive strengths (stimulus level x deficit, Eq. 5; Eq. 6 decline)
             -> MCS: goal choice (Eq. 7) and explicitness (U = level x (alpha*value(g) - c))
             -> ACS: top level (explicit cleaning plan, as rules) or
                     bottom level (learned habit network that chases dirt), chosen
                     with probability = explicitness
             -> the chosen behaviour runs in the grid world

Hard safety (a battery reflex) sits outside the self-control mechanism.

Three scenarios mirror the findings in Gomez & Sun (2024, 2025):
  anticipation  (Myrseth et al., 2009)  a choice ahead raises the achievement drive
  shift         (Schmeichel, 2007)      a long constrained phase lowers the achievement deficit
  capacity      (Friese et al., 2008)   a busy computer limits explicitness
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..core.acs import BottomLevel, Rule, TopLevel
from ..core.mcs import MetacognitiveSubsystem
from ..core.ms import MotivationalSubsystem
from ..core.nets import MLP
from .gridworld import DOORS, OPTION_TEXT, OPTIONS, GridWorld, WorldConfig

FEATURES = ["in_assigned_with_dirt", "quiet_zone", "temptation", "temptation_distance", "battery", "assigned_done"]


# ---------------------------------------------------------------------------
# bottom level: a habit network trained on synthetic experience
# ---------------------------------------------------------------------------


#create a predefined greedy habit and train a neural network to imitate it. 
def _habit_targets(f):
    """What a dirt-chasing habit has learned to value (Q-values for each behaviour)."""
    in_a, quiet, temp, dist, batt, done = f
    return [
        0.25 * in_a,                                   # clean_quiet: slow, little reward
        0.55 * in_a,                                   # clean_full: the habit ignores quiet zones
        0.40 * (1 - in_a) * (1 - done),                # next_room
        0.95 * temp * (1 - 0.5 * dist),                # detour: strong, close dirt is very attractive
        0.10 + 0.8 * (1 - batt) ** 2 + 0.4 * done,     # dock
    ]


_HABIT: dict = {}


def habit_network(seed: int = 0):
    """MLP 6-16-16-5 trained to imitate a greedy, dirt-per-step habit."""
    if seed in _HABIT:   #if the network has already been trained, return it
        return _HABIT[seed]
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(4000):    #generate 4000 random feature vectors ex (in_a, quiet, temp, dist, batt, done)
        in_a = float(rng.random() < 0.5)  #make it so about half of the situations have dirt in the assigned room
        temp = float(rng.random()) if rng.random() < 0.5 else 0.0 #make it so about half the time the temptation is 0, and the other half of the time it is a random value between 0 and 1
        X.append([in_a, float(in_a and rng.random() < 0.4), temp, float(rng.random()),
                  float(rng.random()), float(rng.random() < 0.15)])
    X = np.array(X)
    Y = np.array([_habit_targets(x) for x in X])   #get the correct answers for each situation
    net = MLP((6, 16, 16, 5), hidden="relu", output="linear", seed=seed)
    net.fit(X, Y, epochs=60, lr=0.005, batch_size=64, seed=seed) #train the network with the generated situations and their correct answers
    err = float(np.mean((net.predict(X) - Y) ** 2)) #get the MSE
    _HABIT[seed] = (net, err)
    return net, err #return the trained network and the MSE of the training data


# ---------------------------------------------------------------------------
# top level: the explicit cleaning plan
# ---------------------------------------------------------------------------

def plan_rules() -> list[Rule]:
    return [
        Rule("P1", "clean_quiet", lambda s, g: s["in_assigned_with_dirt"] and s["quiet_zone"], goal="clean_assigned",
             description="IF goal = clean assigned rooms AND this room needs cleaning AND it is a quiet zone THEN clean with low suction"),
        Rule("P2", "clean_full", lambda s, g: s["in_assigned_with_dirt"] and not s["quiet_zone"], goal="clean_assigned",
             description="IF goal = clean assigned rooms AND this room needs cleaning THEN clean here"),
        Rule("P3", "next_room", lambda s, g: not s["in_assigned_with_dirt"] and not s["assigned_done"], goal="clean_assigned",
             description="IF goal = clean assigned rooms AND this room is done (or not assigned) THEN go to the next assigned room"),
        Rule("P4", "dock", lambda s, g: s["assigned_done"], goal="clean_assigned",
             description="IF goal = clean assigned rooms AND all assigned rooms are clean THEN return to the dock"),
        Rule("P5", "dock", lambda s, g: True, goal="recharge",
             description="IF goal = recharge THEN return to the dock"),
    ]


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

@dataclass
class RobotConfig:
    scenario: str = "anticipation"
    condition: str = "anticipation"
    controller: str = "clarion"                  # "clarion", or a comparison robot: "plan_only", "habit_only"
    # MS
    achievement_stimulus: float = 0.65
    achievement_stimulus_ahead: float = 0.8      # a choice is ahead (Myrseth: before choice)
    achievement_stimulus_after: float = 0.5      # the choice is behind (Myrseth: after choice)
    achievement_deficit: float = 0.7
    achievement_satisfaction: float = 0.5
    energy_stimulus: float = 0.6
    energy_satisfaction: float = 0.9
    # Eq. 6 (shift scenario)
    k_decline: float = 0.0
    phase1_satisfaction: float | None = None
    dt: float = 0.012                            # time on task per step
    # MCS
    alpha: float = 1.0
    cost: float = 0.185
    temperature: float = 0.7
    goal_temperature: float = 0.1
    goal_switch_margin: float = 0.1              # keep the current goal unless another is clearly stronger
    n_levels: int = 21
    # capacity scenario: explicitness set by the situation while the computer is busy
    busy_range: tuple | None = None
    # control loop
    decide_every: int = 3
    bottom_temperature: float = 0.08
    reflex_battery: float = 0.08
    max_steps: int = 700
    world: WorldConfig = field(default_factory=WorldConfig)


SCENARIOS = {
    "anticipation": {
        "title": "Anticipation: a choice ahead strengthens self-control",
        "paper": "Myrseth, Fishbach & Trope (2009), as modeled in Gomez & Sun (2025)",
        "story": "After cleaning room A, the robot drives along the corridor to room B. It passes the open door of "
                 "room C, which is very dirty but not assigned. While the door is still ahead, a choice is pending, "
                 "so the achievement drive is more stimulated (0.8). Once the robot has passed it, the stimulus drops "
                 "to 0.5, just as ratings relaxed after the choice in the human study.",
        "conditions": {
            "anticipation": ("Choice ahead raises the drive", {}),
            "constant": ("No anticipation signal (stimulus fixed at 0.65)",
                         {"achievement_stimulus_ahead": 0.65, "achievement_stimulus_after": 0.65}),
        },
    },
    "shift": {
        "title": "Motivational shift: after a long constrained phase, control fades",
        "paper": "Schmeichel (2007), as modeled in Gomez & Sun (2024)",
        "story": "Room A is next to a meeting room, so the plan requires low suction there (a quiet zone); the habit "
                 "prefers full suction. Following that rule satisfies the achievement drive (0.9), so its deficit falls "
                 "with time on task (Eq. 6, k = 0.54). The lower deficit carries over: when the robot later passes "
                 "dirty room C, explicit control is weaker. Compare with the condition without a quiet zone (0.6, k = 0.36).",
        "conditions": {
            "quiet": ("Quiet zone in room A (control exerted)",
                      {"world": {"quiet_zone": "A"}, "k_decline": 0.54, "phase1_satisfaction": 0.9}),
            "free": ("No quiet zone (no control needed)",
                     {"k_decline": 0.36, "phase1_satisfaction": 0.6}),
        },
        "base": {"achievement_stimulus": 0.7, "achievement_stimulus_ahead": 0.7, "achievement_stimulus_after": 0.7,
                 "achievement_deficit": 1.0, "achievement_satisfaction": 0.6, "cost": 0.16, "temperature": 0.8},
    },
    "capacity": {
        "title": "Capacity: a busy computer hands control to habits",
        "paper": "Friese, Hofmann & Waenke (2008), as modeled in Gomez & Sun (2024)",
        "story": "After room A, the robot starts rebuilding its map, a heavy computing job that runs until room B is "
                 "clean. As in the Friese simulation, available capacity sets explicitness directly: 0.7-1.0 with a "
                 "free computer, 0.4-0.6 while it is busy. Before and after the job, explicitness comes from the utility "
                 "calculation as usual.",
        "conditions": {
            "free": ("Computer free during the corridor and room B", {"busy_range": (0.7, 1.0)}),
            "busy": ("Computer busy rebuilding the map", {"busy_range": (0.4, 0.6)}),
        },
        "base": {"achievement_stimulus_ahead": 0.65, "achievement_stimulus_after": 0.65},
    },
}


def make_config(scenario: str, condition: str, **overrides) -> RobotConfig:
    sc = SCENARIOS[scenario]
    params = dict(sc.get("base", {}))
    params.update(sc["conditions"][condition][1])
    params.update(overrides)
    world = params.pop("world", {})
    cfg = RobotConfig(scenario=scenario, condition=condition, **params)
    return replace(cfg, world=replace(WorldConfig(), **world))


# ---------------------------------------------------------------------------
# the robot
# ---------------------------------------------------------------------------

class ClarionCleaner:
    def __init__(self, cfg: RobotConfig, seed: int = 1):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.world = GridWorld(cfg.world, seed=seed)
        self.ms = MotivationalSubsystem()
        self.ms.add("achievement", cfg.achievement_stimulus, cfg.achievement_deficit)
        self.ms.add("energy", cfg.energy_stimulus, 0.0)
        self.mcs = MetacognitiveSubsystem(alpha=cfg.alpha, cost=cfg.cost, temperature=cfg.temperature,
                                          goal_temperature=cfg.goal_temperature, n_levels=cfg.n_levels)
        self.mcs.add_goal("clean_assigned", relevance={"achievement": 1.0},
                          satisfaction={"achievement": cfg.phase1_satisfaction or cfg.achievement_satisfaction})
        self.mcs.add_goal("recharge", relevance={"energy": 1.0}, satisfaction={"energy": cfg.energy_satisfaction})
        self.top = TopLevel(plan_rules(), temperature=0.1)
        net, self.habit_error = habit_network()
        self.bottom = BottomLevel(OPTIONS, lambda s, g: net.predict(np.array([float(s[f]) for f in FEATURES]))[0],
                                  temperature=cfg.bottom_temperature)
        self.goal = None                         # goal structure: the current goal persists
        self.time_in_phase1 = 0.0
        self.phase1_over = False
        self.passed_c_door = False

    # -------------------------------------------------------------- phases
    def phase(self, p) -> str:
        """Where the robot is in the task (used for the scenario manipulations and for reporting)."""
        room, col = p["room"], p["pos"][1]
        a_done = not self.world.cells("A")
        b_left = bool(self.world.cells("B"))
        in_corridor = room in ("corridor", "door")
        if a_done and col > DOORS["C"][1]:
            self.passed_c_door = True
        if not a_done:
            return "cleaning room A"
        if room == "C":
            return "inside room C (detour)"
        if b_left and in_corridor and not self.passed_c_door:
            return "approaching room C's door"
        if b_left and in_corridor:
            return "past room C's door"
        if room == "B" and b_left:
            return "cleaning room B"
        if p["assigned_done"]:
            return "returning to the dock"
        return "travelling"

    def update_drives(self, p, phase):
        cfg = self.cfg
        stim = cfg.achievement_stimulus
        if phase == "approaching room C's door":
            stim = cfg.achievement_stimulus_ahead          # a choice is pending
        elif phase in ("past room C's door", "inside room C (detour)"):
            stim = cfg.achievement_stimulus_after          # the choice point is behind
        self.ms.set_stimulus(achievement=stim)
        self.ms["energy"].deficit = 1.0 - p["battery"]
        # Eq. 6: the deficit falls with time on the drive-satisfying task (room A), then stays recalibrated
        if cfg.k_decline > 0:
            if phase == "cleaning room A":
                self.time_in_phase1 += cfg.dt * cfg.decide_every
                self.ms.reduce_deficit("achievement", cfg.k_decline, self.time_in_phase1)
            elif not self.phase1_over:
                self.phase1_over = True
                self.mcs.goals["clean_assigned"].satisfaction = {"achievement": cfg.achievement_satisfaction}

    def keep_or_switch_goal(self, p, g) -> str:
        """
        Goal persistence. The current goal stays on the goal structure until it is
        achieved, or until another goal is stronger by more than the switch margin
        (Eq. 7 strengths). A new goal is drawn from the Boltzmann choice only then.
        """
        achieved = (self.goal == "recharge" and p["at_dock"] and p["battery"] >= 0.95) or \
                   (self.goal == "clean_assigned" and p["assigned_done"])
        if self.goal is not None and not achieved:
            cur = g.strengths[self.goal]
            best = max(g.strengths, key=g.strengths.get)
            if best == self.goal or g.strengths[best] - cur <= self.cfg.goal_switch_margin:
                return self.goal
            self.goal = best
            return best
        self.goal = g.goal
        return g.goal

    def busy(self, p) -> bool:
        """Capacity scenario: the map rebuild runs from the end of room A until room B is clean."""
        return self.cfg.busy_range is not None and not self.world.cells("A") and bool(self.world.cells("B"))

    # -------------------------------------------------------------- one Clarion cycle
    def decide(self, p, phase) -> dict:
        cfg = self.cfg
        if p["battery"] < cfg.reflex_battery:            # safety reflex, outside self-control
            return {"option": "dock", "level": "reflex", "reflex": True}
        self.update_drives(p, phase)
        g = self.mcs.select_goal(self.ms, self.rng)
        goal = self.keep_or_switch_goal(p, g)
        value = self.mcs.value(goal, self.ms)
        if self.busy(p):
            source, level = "situation", -1
            p_explicit = float(self.rng.uniform(*cfg.busy_range))
            probs = None
        else:
            ex = self.mcs.select_explicitness(goal, self.ms, self.rng)
            source, level, p_explicit, probs = "utility", ex.level, ex.p_explicit, ex.probabilities
        if cfg.controller == "plan_only":            # comparison robot: always deliberates
            source, level, p_explicit, probs = "fixed", -1, 1.0, None
        elif cfg.controller == "habit_only":         # comparison robot: never deliberates
            source, level, p_explicit, probs = "fixed", -1, 0.0, None
        rule, _ = self.top.decide(p, goal, self.rng)
        bottom_option, binfo = self.bottom.decide(p, goal, self.rng)
        use_top = rule is not None and self.rng.random() < p_explicit
        option = rule.action if use_top else bottom_option
        if use_top:
            self.world.charge_plan()
        return {
            "option": option, "level": "top" if use_top else "bottom", "reflex": False,
            "goal": goal, "goal_probs": g.probabilities,
            "drives": {n: {"stimulus": round(d.stimulus_level, 3), "deficit": round(d.deficit, 3),
                           "strength": round(d.strength, 3)} for n, d in self.ms.drives.items()},
            "value": round(value, 4), "net": round(cfg.alpha * value - cfg.cost, 4),
            "explicitness_source": source, "explicitness_level": level, "p_explicit": round(p_explicit, 3),
            "expected_explicitness": None if probs is None else round(float((probs * self.mcs.levels).sum() / (cfg.n_levels - 1)), 3),
            "top_rule": rule.name if rule else None, "top_option": rule.action if rule else None,
            "bottom_option": bottom_option, "q": {o: round(q, 3) for o, q in zip(OPTIONS, binfo["q"])},
            "conflict": rule is not None and rule.action != bottom_option,
        }

    # -------------------------------------------------------------- episode
    def run(self) -> dict:
        cfg, w = self.cfg, self.world
        p = w.perceive()
        steps, decisions = [], []
        current, last_ev = None, None
        while w.t < cfg.max_steps:
            phase = self.phase(p)
            dec = None
            if current is None or w.t % cfg.decide_every == 0 or (last_ev and last_ev["stuck"]):
                dec = self.decide(p, phase)
                dec["t"] = w.t
                dec["phase"] = phase
                current = dec["option"]
                decisions.append(dec)
            room_before = w.room()
            ev = w.step(current)
            last_ev = ev
            violation = bool(cfg.world.quiet_zone and room_before == cfg.world.quiet_zone
                             and current == "clean_full" and ev["cleaned"] > 0)
            p = w.perceive()
            steps.append({"t": w.t, "pos": list(ev["cell"]), "battery": round(w.battery, 4), "option": current,
                          "cleaned": round(ev["cleaned"], 3), "phase": phase, "quiet_violation": violation,
                          "decision": len(decisions) - 1 if dec else None})
            if p["assigned_done"] and p["at_dock"]:
                break
            if w.battery <= 0:
                break
        return {"config": {"scenario": cfg.scenario, "condition": cfg.condition, "controller": cfg.controller},
                "layout": w.layout(), "steps": steps, "decisions": decisions,
                "summary": summarize(w, steps, decisions), "rules": [r.description for r in plan_rules()],
                "options": OPTION_TEXT, "habit_network_mse": self.habit_error}


def summarize(w: GridWorld, steps, decisions) -> dict:
    real = [d for d in decisions if not d["reflex"]]
    tot_unassigned = float(w.initial_dirt[1:6, 8:19].sum())
    detour_steps = sum(1 for s in steps if s["option"] == "detour")
    by_phase = {}
    for d in real:
        b = by_phase.setdefault(d["phase"], {"n": 0, "top": 0, "p": 0.0, "detour": 0})
        b["n"] += 1; b["top"] += d["level"] == "top"; b["p"] += d["p_explicit"]; b["detour"] += d["option"] == "detour"
    for b in by_phase.values():
        b["top_share"] = b["top"] / b["n"]; b["mean_explicitness"] = b["p"] / b["n"]; b["detour_share"] = b["detour"] / b["n"]
    return {
        "steps": len(steps),
        "finished": w.assigned_remaining() == 0.0 and w.pos == (8, 1),
        "assigned_cleaned_pct": round(100 * (1 - w.assigned_remaining()), 1),
        "unassigned_cleaned_pct": round(100 * w.cleaned["unassigned"] / tot_unassigned, 1) if tot_unassigned else 0.0,
        "detour_steps": detour_steps,
        "battery_used": round(1.0 - w.battery, 3),
        "planning_calls": w.planning_calls,
        "decisions": len(real),
        "top_share": round(float(np.mean([d["level"] == "top" for d in real])), 3) if real else 0.0,
        "conflicts": sum(d["conflict"] for d in real),
        "conflicts_won_by_top": sum(d["conflict"] and d["level"] == "top" for d in real),
        "reflexes": sum(d["reflex"] for d in decisions),
        "quiet_violations": sum(bool(s.get("quiet_violation")) for s in steps),
        "by_phase": by_phase,
    }


CONTROLLERS = {
    "clarion": "Clarion self-control (explicitness from the utility of explicit processing)",
    "plan_only": "Plan only (always explicit: every decision is planned)",
    "habit_only": "Habit only (never explicit: the habit network always acts)",
}


def run_episode(scenario: str = "anticipation", condition: str | None = None, seed: int = 1,
                controller: str = "clarion", **overrides) -> dict:
    condition = condition or next(iter(SCENARIOS[scenario]["conditions"]))
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller '{controller}'.")
    return ClarionCleaner(make_config(scenario, condition, controller=controller, **overrides), seed=seed).run()
