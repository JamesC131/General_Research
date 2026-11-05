"""
Simulation 1: Schmeichel (2007), self-control and working memory.
Source: Gomez & Sun (2024), "A 'Rational' Framework for Self-Control".

Activity 1: watch a video with distractor words, either ignoring the words
(attention control) or watching freely (no control).
Activity 2: an operation-span test (Test 1: 41 vs 38 participants) or a
sentence-span test (Test 2: 31 vs 31). Each test: 15 sets of 2-5 items.

Mechanism (motivational shift): exerting control satisfies the achievement
drive more (0.9 vs 0.6), and its deficit falls faster (Eq. 6; k = 0.54 vs 0.36).
Lower drive strength -> lower value(g) -> lower utility of explicit processing
-> the ACS shifts toward the bottom level. In Activity 2 the top level alone
controls encoding into the LSTM working memory, so memory scores drop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..agent import ClarionAgent
from ..analysis import stats as S
from ..core.acs import ActionCenteredSubsystem, BottomLevel, Rule, TopLevel, make_mlp_q
from ..core.mcs import MetacognitiveSubsystem
from ..core.ms import MotivationalSubsystem
from ..core.nacs import WorkingMemory
from ..core.nets import MLP
from .common import N_LEVELS, P

VIDEO_ACTIONS = ["do_nothing", "read_words_without_watching", "watch_video_and_read_words",
                 "watch_video_and_do_not_read_words"]

HUMAN = {  # Schmeichel (2007), as reported in Gomez & Sun (2024)
    "test1": {"attention_control": 32.88, "no_control": 34.68, "n": (41, 38)},
    "test2": {"attention_control": 34.87, "no_control": 38.32, "n": (31, 31)},
}
PUBLISHED_SIM = {
    "test1": {"attention_control": 31.8, "no_control": 35.1},
    "test2": {"attention_control": 33.5, "no_control": 38.0},
}
_CAL_PATH = Path(__file__).resolve().parent.parent / "data" / "schmeichel_calibration.json"

# 15 sets of 2-5 items: 48 items (operation span), 50 items (sentence span)
SETS_TEST1 = [2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5]
SETS_TEST2 = [2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5]


@dataclass
class SchmeichelConfig:
    n_test1: tuple = P((41, 38), "paper", "(attention control, no control)")
    n_test2: tuple = P((31, 31), "paper")
    # MS
    achievement_stimulus: float = P(0.7, "paper")
    initial_deficit: float = P(1.0, "paper")
    deficit_sd: float = P(0.25, "calibrated",
                          "individual differences in the starting achievement deficit (SD); deficit = individual predisposition")
    k_attention: float = P(0.54, "paper", "Eq. 6 scaling, attention-control group")
    k_no_control: float = P(0.36, "paper", "Eq. 6 scaling, no-control group")
    satisfaction_attention: float = P(0.9, "paper", "achievement satisfaction in Activity 1")
    satisfaction_no_control: float = P(0.6, "paper")
    satisfaction_memory: float = P(0.6, "chosen", "achievement satisfaction of the memory task (same for both groups)")
    # MCS
    alpha: float = P(1.0, "paper")
    cost: float = P(0.16, "paper")
    temperature: float = P(0.8, "paper")
    n_levels: int = P(N_LEVELS, "chosen")
    # Activity 1 (video)
    video_steps: int = P(30, "chosen", "decision cycles during the video")
    time_on_task: float = P(0.7, "calibrated", "total Activity-1 time in the units of Eq. 6")
    word_probability: float = P(0.5, "chosen", "share of video moments with words on screen")
    k_jitter: float = P(0.05, "chosen", "individual differences in k (SD)")
    # Activity 2 (memory, LSTM in the NACS)
    lstm_temperature: float = P(1.5, "calibrated", "recall noise of the LSTM")
    implicit_gain_test1: float = P(0.2, "calibrated", "encoding strength without top-level control, Test 1")
    implicit_gain_test2: float = P(0.4, "calibrated", "encoding strength without top-level control, Test 2")
    distractors_test1: int = P(1, "chosen", "processing steps between items (equations)")
    distractors_test2: int = P(1, "chosen", "processing steps between items (sentences)")


def default_config(**overrides) -> SchmeichelConfig:
    base = json.loads(_CAL_PATH.read_text()) if _CAL_PATH.exists() else {}
    base.update(overrides)
    return SchmeichelConfig(**base)


# ---------------------------------------------------------------------------
# Activity 1 bottom level: 4 inputs, 8 hidden, 4 outputs (backpropagation)
# ---------------------------------------------------------------------------

def _video_tendency(watch_freely, ignore_words, intensity, distraction):
    """Synthetic 'participant behaviour' used to train the bottom-level network (Q-values)."""
    return np.array([
        0.05,
        0.10 + 0.45 * distraction * (1 - intensity),
        0.15 + 0.55 * distraction * intensity + 0.15 * watch_freely,
        0.20 + 0.60 * intensity * (1 - distraction) + 0.20 * ignore_words,
    ])


_VIDEO_NET: dict = {}


def video_network(seed: int = 0) -> tuple[MLP, dict]:
    if seed in _VIDEO_NET:
        return _VIDEO_NET[seed]
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(1200):
        free = float(rng.random() < 0.5)
        X.append([free, 1 - free, rng.random(), rng.random() * (rng.random() < 0.6)])
    X = np.array(X)
    Y = np.array([_video_tendency(*x) for x in X]) + rng.normal(0, 0.03, (len(X), 4))
    tr, va = slice(0, 1000), slice(1000, None)
    net = MLP((4, 8, 4), hidden="tanh", output="sigmoid", seed=seed)
    net.fit(X[tr], np.clip(Y[tr], 0, 1), epochs=150, lr=0.01, batch_size=50, seed=seed)
    pred = net.predict(X[va])
    target = np.array([_video_tendency(*x) for x in X[va]])
    info = {"validation_mse": float(np.mean((pred - target) ** 2)),
            "validation_argmax_agreement": float(np.mean(pred.argmax(1) == target.argmax(1)))}
    _VIDEO_NET[seed] = (net, info)
    return net, info


def video_rules() -> list[Rule]:
    return [
        Rule("RI1", "watch_video_and_do_not_read_words",
             lambda s, g: s["ignore_words"] and s["words_on_screen"], goal="participate",
             description="IF (instruction = ignore words AND words on screen) THEN watch the video and do not read the words"),
        Rule("RI2", "watch_video_and_do_not_read_words",
             lambda s, g: s["ignore_words"] and not s["words_on_screen"], goal="participate",
             description="IF (instruction = ignore words AND no words on screen) THEN keep watching the video"),
        Rule("RI3", "watch_video_and_read_words",
             lambda s, g: s["watch_freely"], goal="participate",
             description="IF (instruction = watch freely) THEN watch the video (reading words is allowed)"),
    ]


def memory_rules() -> list[Rule]:
    return [
        Rule("MU1", "encode_item", lambda s, g: s.get("phase") == "encode", goal="participate",
             description="IF (item presented AND goal = perform task) THEN update working memory with the item"),
        Rule("MU2", "recall_in_order", lambda s, g: s.get("phase") == "recall", goal="participate",
             description="IF (recall cue) THEN retrieve the items from working memory in order"),
    ]


# ---------------------------------------------------------------------------
# participant
# ---------------------------------------------------------------------------

def run_participant(cfg: SchmeichelConfig, group: str, test: str, seed: int, wm: WorkingMemory) -> dict:
    rng = np.random.default_rng(seed)
    attention = group == "attention_control"
    k = max(0.0, (cfg.k_attention if attention else cfg.k_no_control) + rng.normal(0, cfg.k_jitter))

    ms = MotivationalSubsystem()
    ms.add("achievement", cfg.achievement_stimulus,
           float(np.clip(cfg.initial_deficit + rng.normal(0, cfg.deficit_sd), 0.05, 1.5)))
    mcs = MetacognitiveSubsystem(alpha=cfg.alpha, cost=cfg.cost, temperature=cfg.temperature,
                                 n_levels=cfg.n_levels)
    mcs.add_goal("participate", relevance={"achievement": 1.0},
                 satisfaction={"achievement": cfg.satisfaction_attention if attention else cfg.satisfaction_no_control})

    vnet, _ = video_network()

    def encode(state, goal):
        return np.array([float(state["watch_freely"]), float(state["ignore_words"]),
                         state["intensity"], state["distraction"]])

    video_acs = ActionCenteredSubsystem(TopLevel(video_rules()),
                                        BottomLevel(VIDEO_ACTIONS, make_mlp_q(vnet, encode), temperature=0.1))
    agent = ClarionAgent(ms, mcs, video_acs, nacs=wm, seed=seed + 1)

    # ---- Activity 1: video ------------------------------------------------------------
    timeline = []
    for step in range(cfg.video_steps):
        t = (step + 1) / cfg.video_steps * cfg.time_on_task
        ms.reduce_deficit("achievement", k, t)                 # Eq. 6
        words = rng.random() < cfg.word_probability
        state = {"watch_freely": not attention, "ignore_words": attention, "words_on_screen": words,
                 "intensity": float(rng.uniform(0.3, 0.9)), "distraction": float(rng.uniform(0.4, 1.0)) if words else 0.0}
        rec = agent.decide({"state": state, "goal": "participate"})
        timeline.append({"t": t, "deficit": ms["achievement"].deficit, "drive": ms["achievement"].strength,
                         "value": rec.value, "p_explicit": rec.p_explicit, "level": rec.level_used,
                         "action": rec.action, "words": words})
    failures = [e["action"] in ("read_words_without_watching", "watch_video_and_read_words")
                for e in timeline if e["words"]]

    # ---- Activity 2: memory test (deficit stays recalibrated) ------------------------
    mcs.goals["participate"].satisfaction = {"achievement": cfg.satisfaction_memory}
    mem_acs = ActionCenteredSubsystem(TopLevel(memory_rules()),
                                      BottomLevel(["attend_passively"], lambda s, g: np.zeros(1)))
    sets = SETS_TEST1 if test == "test1" else SETS_TEST2
    gain = cfg.implicit_gain_test1 if test == "test1" else cfg.implicit_gain_test2
    dist = cfg.distractors_test1 if test == "test1" else cfg.distractors_test2
    wm.temperature, wm.implicit_gain = cfg.lstm_temperature, gain
    score, mem_levels = 0, []
    for size in sets:
        digits = rng.integers(0, 10, size)
        controlled = []
        for _ in range(size):
            ex = mcs.select_explicitness("participate", ms, agent.rng)
            dec = mem_acs.decide({"phase": "encode"}, "participate", ex.p_explicit, agent.rng)
            controlled.append(dec.action == "encode_item")
            mem_levels.append(ex.p_explicit)
        score += wm.score(digits, controlled, agent.rng, distractors_between=dist)

    return {"group": group, "test": test, "score": int(score), "max_score": int(sum(sets)), "k": k,
            "final_deficit": ms["achievement"].deficit,
            "explicitness_activity1_start": float(np.mean([e["p_explicit"] for e in timeline[:5]])),
            "explicitness_activity1_end": float(np.mean([e["p_explicit"] for e in timeline[-5:]])),
            "explicitness_activity2": float(np.mean(mem_levels)),
            "control_failures": float(np.mean(failures)) if failures else float("nan"),
            "timeline": timeline}


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def run(cfg: SchmeichelConfig | None = None, seed: int = 1, wm: WorkingMemory | None = None) -> dict:
    cfg = cfg or default_config()
    wm = wm or WorkingMemory.load()
    parts, base = [], seed * 100003
    for test, (na, nn) in (("test1", cfg.n_test1), ("test2", cfg.n_test2)):
        for group, n in (("attention_control", na), ("no_control", nn)):
            for i in range(n):
                base += 1
                parts.append(run_participant(cfg, group, test, base, wm))
    return {"config": cfg, "participants": parts, "stats": analyze(parts), "video_network": video_network()[1]}


def analyze(parts: list[dict]) -> dict:
    out = {}
    for test in ("test1", "test2"):
        a = [p["score"] for p in parts if p["test"] == test and p["group"] == "attention_control"]
        n = [p["score"] for p in parts if p["test"] == test and p["group"] == "no_control"]
        out[test] = {"attention_control": S.describe(a), "no_control": S.describe(n),
                     "test": S.independent_t(a, n)}
    for group in ("attention_control", "no_control"):
        g = [p for p in parts if p["group"] == group]
        out[group] = {k: float(np.nanmean([p[k] for p in g])) for k in
                      ("explicitness_activity1_start", "explicitness_activity1_end",
                       "explicitness_activity2", "control_failures", "final_deficit")}
    return out
