"""
Bridge for the native Soar self-control model (native_cleaner.soar).

NativeSoarCleaner reuses pyclarion-sc's cleaning-robot world and episode loop
(ClarionCleaner.run), so all three architectures meet exactly the same world,
phases, safety reflex and planning cost. Only `decide()` differs.

Native mechanisms and what the bridge supplies:
  * habit:     Soar-RL; the bridge writes the reward (dirt cleaned since the last decision)
  * features:  discretized perception (Soar-RL needs symbolic situations)
  * capacity:  a budget of Soar decision cycles per robot cycle. When it is used up,
               the bridge sets ^time-up true and the robot must act with what it has.

Typical use:
    from native_soar.native_cleaner import train_habit, run_native_episode
    train_habit()                                  # once; writes native_soar/habit_rl.soar
    ep = run_native_episode("capacity", "busy", seed=1)
"""
from __future__ import annotations

import os

import numpy as np
import soar_sml as sml

from pyclarion_sc.robot.controller import ClarionCleaner, make_config

HERE = os.path.dirname(os.path.abspath(__file__))
SOAR_FILE = os.path.join(HERE, "native_cleaner.soar")
HABIT_FILE = os.path.join(HERE, "habit_rl.soar")

# ----------------------------------------------------------------- parameters (chosen; see Document 1)
NATIVE = {
    "temptation_low_high": 0.5,     # temptation-level: none / low (<0.5) / high
    "near_distance": 0.25,          # temptation-dist: near if distance <= 0.25 (5 cells)
    "battery_low": 0.25, "battery_mid": 0.6,
    "test_temperature": 0.1,        # Boltzmann temperature for the trained habit
    "train_temperature": 0.3,
    "learning_rate": 0.1, "discount": 0.9,
    "train_episodes": 300, "train_seed0": 10001,
    "budget_free": None,            # no time limit
    # capacity scenario: decision cycles available per robot cycle (calibrated on dev seeds, then frozen)
    "budget_busy": (3, 8),          # uniform integer in [lo, hi]
    "budget_capacity_free": (12, 16),
}
_KERNEL = None


def kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = sml.Kernel.CreateKernelInCurrentThread(True, 0)
    return _KERNEL


def tf(x):
    return "true" if x else "false"


class NativeSoarCleaner(ClarionCleaner):
    MAX_DECISIONS = 60

    def __init__(self, cfg, seed=1, mode="native", chunking=True, learning=False, habit_file=HABIT_FILE,
                 temperature=None, budget=None, params=None):
        """
        mode:      "native" (deliberate on conflict), "always" (deliberate every cycle),
                   "habit" (never deliberate)
        chunking:  compile deliberations into rules (ablation: False)
        learning:  Soar-RL learning on (training) or off (test)
        budget:    None = scenario default; or a fixed int; or (lo, hi)
        """
        super().__init__(cfg, seed=seed)
        self.P = dict(NATIVE, **(params or {}))
        self.mode, self.learning = mode, learning
        self.budget_spec = budget
        self.agent = kernel().CreateAgent(f"native_{id(self)}")
        cmds = [f"source {{{SOAR_FILE}}}"]
        if habit_file and os.path.exists(habit_file):
            cmds.append(f"source {{{habit_file}}}")
        cmds += ["rl --set learning " + ("on" if learning else "off"),
                 f"rl --set learning-rate {self.P['learning_rate']}",
                 f"rl --set discount-rate {self.P['discount']}",
                 "decide indifferent-selection --boltzmann",
                 f"decide indifferent-selection --temperature "
                 f"{temperature or (self.P['train_temperature'] if learning else self.P['test_temperature'])}",
                 "chunk " + ("always" if chunking else "never"),
                 f"srand {seed}", "watch 0", "warnings --off"]
        for c in cmds:
            out = self.agent.ExecuteCommandLine(c)
            if not self.agent.GetLastCommandLineResult():
                raise RuntimeError(f"{c}\n{out}")
        a, il = self.agent, self.agent.GetInputLink()
        a.CreateStringWME(il, "mode", mode)
        a.CreateStringWME(il, "learning", "on" if learning else "off")
        self.w = {k: a.CreateStringWME(il, k, v) for k, v in [
            ("in-assigned-with-dirt", "false"), ("quiet-zone", "false"), ("temptation-level", "none"),
            ("temptation-dist", "far"), ("battery-level", "high"), ("assigned-done", "false"),
            ("at-dock", "false"), ("time-up", "false")]}
        self.w_reward = a.CreateFloatWME(il, "reward", 0.0)
        self.w_cycle = a.CreateIntWME(il, "cycle", -1)
        a.Commit()
        self.n = 0
        self._cleaned_before = 0.0

    def close(self):
        kernel().DestroyAgent(self.agent)

    # ---------------------------------------------------------------- perception
    def features(self, p):
        P = self.P
        t = p["temptation"]
        b = p["battery"]
        return {
            "in-assigned-with-dirt": tf(p["in_assigned_with_dirt"]), "quiet-zone": tf(p["quiet_zone"]),
            "temptation-level": "none" if t <= 0 else ("low" if t < P["temptation_low_high"] else "high"),
            "temptation-dist": "near" if p["temptation_distance"] <= P["near_distance"] else "far",
            "battery-level": "low" if b < P["battery_low"] else ("mid" if b < P["battery_mid"] else "high"),
            "assigned-done": tf(p["assigned_done"]), "at-dock": tf(p["at_dock"]),
        }

    def budget(self, p):
        spec = self.budget_spec
        if spec is None:
            if self.cfg.busy_range is not None and self.busy(p):
                spec = self.P["budget_busy"] if self.cfg.busy_range[1] <= 0.6 else self.P["budget_capacity_free"]
            else:
                spec = self.P["budget_free"]
        if spec is None:
            return self.MAX_DECISIONS
        if isinstance(spec, int):
            return spec
        return int(self.rng.integers(spec[0], spec[1] + 1))

    def _read(self):
        """The command for this cycle, merged with this cycle's reports (controlled / deliberated)."""
        ol = self.agent.GetOutputLink()
        if ol is None:
            return None
        cmd, extra = None, {}
        for i in range(ol.GetNumberChildren()):
            ch = ol.GetChild(i)
            if ch.GetAttribute() in ("command", "report") and ch.IsIdentifier():
                c = ch.ConvertToIdentifier()
                v = {c.GetChild(j).GetAttribute(): c.GetChild(j).GetValueAsString() for j in range(c.GetNumberChildren())}
                if v.get("cycle") == str(self.n):
                    if ch.GetAttribute() == "command":
                        cmd = v
                    else:
                        extra.update(v)
        if cmd is not None:
            cmd.update({k: v for k, v in extra.items() if k != "cycle"})
        return cmd

    # ---------------------------------------------------------------- one decision
    def decide(self, p, phase):
        if p["battery"] < self.cfg.reflex_battery:
            return {"option": "dock", "level": "reflex", "reflex": True}
        a = self.agent
        self.n += 1
        cleaned = self.world.cleaned["assigned"] + self.world.cleaned["unassigned"]
        a.Update(self.w_reward, float(cleaned - self._cleaned_before))
        self._cleaned_before = cleaned
        for k, v in self.features(p).items():
            a.Update(self.w[k], v)
        a.Update(self.w["time-up"], "false")
        a.Update(self.w_cycle, self.n)
        a.Commit()
        budget = self.budget(p)
        vals, used, timed_out = None, 0, False
        while used < self.MAX_DECISIONS:
            if used >= budget and not timed_out:
                a.Update(self.w["time-up"], "true")
                a.Commit()
                timed_out = True
            a.RunSelf(1)
            used += 1
            vals = self._read()
            if vals:
                break
        if vals is None:
            raise RuntimeError("no command\n" + a.ExecuteCommandLine("print --stack"))
        deliberated = vals.get("deliberated") == "true"
        controlled = vals.get("controlled") == "true"
        if deliberated:
            self.world.charge_plan()            # same battery cost of explicit processing as CLARION
        return {
            "option": vals["option"], "level": "top" if controlled else "bottom", "reflex": False,
            "goal": "clean_assigned", "p_explicit": float(controlled), "explicitness_source": "native",
            "explicitness_level": -1, "value": float("nan"), "drives": {}, "q": {},
            "top_rule": None, "bottom_option": None, "conflict": deliberated,
            "decisions_used": used, "budget": budget, "timed_out": timed_out and not controlled,
            "deliberated": deliberated,
        }


# ---------------------------------------------------------------------- training the habit
def train_habit(episodes=None, out=HABIT_FILE, verbose=True):
    """Soar-RL training of the habit: reward = dirt cleaned (any room); no deliberation."""
    episodes = episodes or NATIVE["train_episodes"]
    if os.path.exists(out):
        os.remove(out)
    robot = None
    for i in range(episodes):
        quiet = {"world": {"quiet_zone": "A"}} if i % 2 else {}
        cfg = make_config("anticipation", "constant")
        if quiet:
            from dataclasses import replace
            cfg = replace(cfg, world=replace(cfg.world, quiet_zone="A"))
        if robot is None:
            robot = NativeSoarCleaner(cfg, seed=NATIVE["train_seed0"] + i, mode="habit", chunking=False,
                                      learning=True, habit_file=None)
        else:   # keep the same agent (and its learned values); new world
            from pyclarion_sc.robot.gridworld import GridWorld
            robot.cfg = cfg
            robot.world = GridWorld(cfg.world, seed=NATIVE["train_seed0"] + i)
            robot._cleaned_before = 0.0
            robot.time_in_phase1, robot.phase1_over, robot.passed_c_door = 0.0, False, False
        robot.run()
        if verbose and (i + 1) % 50 == 0:
            print(f"  habit training: {i + 1}/{episodes} episodes")
    text = robot.agent.ExecuteCommandLine("print --rl --full")
    robot.close()
    with open(out, "w") as f:
        f.write("# Soar-RL habit values learned by train_habit() (reward = dirt cleaned, no deliberation)\n")
        f.write(text)
    return out


def habit_table(habit_file=HABIT_FILE):
    """Learned values as {(situation...), behaviour: value} for inspection."""
    import re
    rows = []
    txt = open(habit_file).read()
    for m in re.finditer(r"sp \{(rl\*rl\*habit\*\d+)(.*?)\n\}", txt, re.S):
        lhs, rhs = m.group(2).split("-->")
        cond = dict(re.findall(r"\^([\w-]+) ([\w-]+)", lhs))
        cond.pop("stage", None); cond.pop("behaviour", None); cond.pop("io", None); cond.pop("input-link", None)
        val = float(re.search(r"=\s*([-\d.eE]+)", rhs).group(1))
        rows.append((cond, val))
    return rows


def run_native_episode(scenario="anticipation", condition=None, seed=1, **kw):
    from pyclarion_sc.robot.controller import SCENARIOS
    condition = condition or next(iter(SCENARIOS[scenario]["conditions"]))
    robot = NativeSoarCleaner(make_config(scenario, condition), seed=seed, **kw)
    try:
        ep = robot.run()
    finally:
        robot.close()
    ep["config"]["architecture"] = f"native soar ({kw.get('mode', 'native')})"
    return ep
