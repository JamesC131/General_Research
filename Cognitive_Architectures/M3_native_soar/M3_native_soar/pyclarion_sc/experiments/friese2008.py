"""
Simulation 2: Friese, Hofmann & Waenke (2008), explicit and implicit attitudes
in food choice. Source: Gomez & Sun (2024), "A 'Rational' Framework for
Self-Control".

85 participants give an explicit attitude rating (fruit vs chocolate), take an
implicit association test, memorize 1 digit (high capacity, n = 42) or 8 digits
(low capacity, n = 43), and choose 5 of 20 items (10 fruits, 10 chocolates).

Human result: the explicit measure predicts the number of chocolates under
high capacity; the implicit measure predicts it under low capacity (Table 1,
Figure 1).

In the model, capacity sets explicitness directly (high: 0.7-1.0, low:
0.4-0.6). On each pick, the ACS top level (explicit attitude, healthy-leaning
rules) acts with probability = explicitness; otherwise the bottom-level
networks (implicit attitude + choice network) act.
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
from ..core.nacs import WorkingMemory
from ..core.nets import MLP
from .common import N_LEVELS, P

FRUITS = [f"fruit_{i}" for i in range(10)]
CHOCOLATES = [f"chocolate_{i}" for i in range(10)]
ITEMS = FRUITS + CHOCOLATES

HUMAN = {  # Table 1 (Friese et al., 2008)
    "high": {"n": 42, "r_exp_imp": 0.20, "r_exp_choc": 0.60, "r_imp_choc": 0.12},
    "low": {"n": 43, "r_exp_imp": 0.37, "r_exp_choc": 0.24, "r_imp_choc": 0.45},
}
PUBLISHED_SIM = {
    "high": {"r_exp_imp": 0.20, "r_exp_choc": 0.60, "r_imp_choc": 0.11},
    "low": {"r_exp_imp": 0.36, "r_exp_choc": 0.23, "r_imp_choc": 0.45},
}
_CAL_PATH = Path(__file__).resolve().parent.parent / "data" / "friese_calibration.json"


@dataclass
class FrieseConfig:
    n_high: int = P(42, "paper")
    n_low: int = P(43, "paper")
    high_explicitness: tuple = P((0.7, 1.0), "paper", "explicitness range under high capacity")
    low_explicitness: tuple = P((0.4, 0.6), "paper", "explicitness range under low capacity")
    digits_high: int = P(1, "paper")
    digits_low: int = P(8, "paper")
    n_picks: int = P(5, "paper")
    # MS / MCS (as in simulation 1; explicitness is set by the situation here)
    achievement_stimulus: float = P(0.7, "paper")
    achievement_deficit: float = P(1.0, "paper")
    # Individual differences (not published; calibrated to Table 1)
    attitude_correlation: float = P(0.40, "calibrated", "correlation of latent explicit and implicit attitudes")
    explicit_slope: float = P(1.2, "calibrated", "how strongly the explicit attitude sets top-level rule support")
    explicit_bias: float = P(-0.6, "calibrated", "health preference of the top-level rules (negative = favors fruit)")
    implicit_mean: float = P(0.4, "calibrated", "mean implicit chocolate preference (logit); paper: network prefers chocolates")
    implicit_sd: float = P(1.2, "calibrated")
    explicit_measure_noise: float = P(0.6, "calibrated", "rating noise of the explicit measure")
    implicit_measure_noise: float = P(0.35, "calibrated", "noise of the implicit (IAT) measure")
    top_temperature: float = P(1.0, "chosen")
    bottom_temperature: float = P(0.25, "chosen")
    attitude_hidden: int = P(3, "paper", "implicit-attitude network: 2 inputs, 3 hidden, 2 outputs")
    choice_hidden: tuple = P((16, 8), "chosen", "choice network: two hidden layers (sizes not given)")
    use_working_memory: bool = P(True, "paper", "digit memorization with the LSTM (manipulation check)")
    train_networks: bool = P(True, "chosen")


def default_config(**overrides) -> FrieseConfig:
    base = json.loads(_CAL_PATH.read_text()) if _CAL_PATH.exists() else {}
    base.update(overrides)
    return FrieseConfig(**base)


def _logit_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------

def train_attitude_network(choc_pref: float, cfg: FrieseConfig, seed: int):
    """Implicit attitude: inputs (fruit present, chocolate present) -> preference (fruit, chocolate)."""
    X = np.array([[1, 0], [0, 1], [1, 1]], float)
    Y = np.array([[1 - choc_pref, choc_pref]] * 3)
    if not cfg.train_networks:
        return _Const(Y[0])
    net = MLP((2, cfg.attitude_hidden, 2), hidden="tanh", output="sigmoid", seed=seed)
    net.fit(X, Y, epochs=400, lr=0.05, seed=seed)
    return net


class _Const:
    def __init__(self, y):
        self.y = np.atleast_2d(y)

    def predict(self, x):
        return np.repeat(self.y, np.atleast_2d(x).shape[0], 0)


_CHOICE_NET: dict = {}


def choice_network(cfg: FrieseConfig):
    """
    Shared bottom-level choice network: 20 availability inputs + the 2 implicit
    preference outputs -> two hidden layers -> 20 item Q-values. Trained once
    on synthetic data (Q of an item = implicit preference for its category).
    """
    key = (cfg.choice_hidden, cfg.train_networks)
    if key in _CHOICE_NET:
        return _CHOICE_NET[key]
    if not cfg.train_networks:
        _CHOICE_NET[key] = None
        return None
    rng = np.random.default_rng(11)
    X, Y = [], []
    for _ in range(3000):
        avail = (rng.random(20) < rng.uniform(0.6, 1.0)).astype(float)
        c = rng.random()
        X.append(np.concatenate([avail, [1 - c, c]]))
        Y.append(np.concatenate([np.full(10, 1 - c), np.full(10, c)]) * avail)
    net = MLP((22, *cfg.choice_hidden, 20), hidden="relu", output="linear", seed=3)
    net.fit(np.array(X), np.array(Y), epochs=60, lr=0.005, batch_size=64, seed=3)
    _CHOICE_NET[key] = net
    return net


# ---------------------------------------------------------------------------
# participant
# ---------------------------------------------------------------------------

def run_participant(cfg: FrieseConfig, capacity: str, seed: int, wm: WorkingMemory | None) -> dict:
    rng = np.random.default_rng(seed)
    rho = cfg.attitude_correlation
    z1, z2 = rng.standard_normal(2)
    E = z1                                    # latent explicit attitude (pro-chocolate)
    I = rho * z1 + np.sqrt(1 - rho**2) * z2   # latent implicit attitude (pro-chocolate)
    choc_pref = float(_logit_sigmoid(cfg.implicit_mean + cfg.implicit_sd * I))

    att_net = train_attitude_network(choc_pref, cfg, seed + 1)
    pref = att_net.predict([[1, 1]])[0]       # (fruit, chocolate) implicit preference
    cnet = choice_network(cfg)

    # --- MS / MCS: goal "pick items"; explicitness set by cognitive capacity --------
    ms = MotivationalSubsystem()
    ms.add("achievement", cfg.achievement_stimulus, cfg.achievement_deficit)
    mcs = MetacognitiveSubsystem(n_levels=N_LEVELS)
    mcs.add_goal("pick_items", relevance={"achievement": 1.0})
    lo, hi = cfg.high_explicitness if capacity == "high" else cfg.low_explicitness
    explicitness = float(rng.uniform(lo, hi))

    # --- ACS ------------------------------------------------------------------
    support_choc = cfg.explicit_bias + cfg.explicit_slope * E
    top = TopLevel(temperature=cfg.top_temperature)
    for item in ITEMS:
        is_choc = item.startswith("chocolate")
        top.add(Rule(f"pick_{item}", f"pick:{item}",
                     condition=(lambda it: (lambda s, g: s["available"][it]))(item),
                     goal="pick_items", support=support_choc if is_choc else 0.0,
                     description=f"IF (goal = pick items AND {item} available) THEN pick {item}"
                                 + (" [explicit valuation of chocolate]" if is_choc else " [healthy item]")))

    def q(state, goal):
        avail = np.array([state["available"][it] for it in ITEMS], float)
        if cnet is None:
            qv = np.concatenate([np.full(10, pref[0]), np.full(10, pref[1])])
        else:
            qv = cnet.predict(np.concatenate([avail, pref]))[0]
        return np.where(avail > 0, qv, -1e9)

    bottom = BottomLevel([f"pick:{it}" for it in ITEMS], q, temperature=cfg.bottom_temperature)
    agent = ClarionAgent(ms, mcs, ActionCenteredSubsystem(top, bottom), seed=seed + 2)

    # --- measures -------------------------------------------------------------------
    explicit_measure = support_choc + rng.normal(0, cfg.explicit_measure_noise)          # top-level evaluation
    implicit_measure = (pref[1] - pref[0]) + rng.normal(0, cfg.implicit_measure_noise)  # IAT-like score

    # --- digit memorization (NACS top-level rule triggers LSTM encoding) -------------
    n_digits = cfg.digits_high if capacity == "high" else cfg.digits_low
    digits = rng.integers(0, 10, n_digits)

    # --- choice task ----------------------------------------------------------------------
    state = {"available": {it: True for it in ITEMS}}
    picks, levels = [], []
    for _ in range(cfg.n_picks):
        rec = agent.decide({"state": state, "goal": "pick_items", "explicitness": explicitness})
        item = rec.action.split(":", 1)[1]
        state["available"][item] = False
        picks.append(item)
        levels.append(rec.level_used)

    recall = wm.score(digits, [True] * n_digits, agent.rng, distractors_between=0) / n_digits if wm else float("nan")

    return {"condition": capacity, "explicitness": explicitness,
            "explicit_measure": float(explicit_measure), "implicit_measure": float(implicit_measure),
            "chocolates": int(sum(p.startswith("chocolate") for p in picks)),
            "picks": picks, "levels": levels, "top_level_share": float(np.mean([l == "top" for l in levels])),
            "digits": n_digits, "digit_recall": recall,
            "latent_explicit": float(E), "latent_implicit": float(I), "implicit_choc_pref": float(pref[1])}


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def run(cfg: FrieseConfig | None = None, seed: int = 1, wm: WorkingMemory | None = None) -> dict:
    cfg = cfg or default_config()
    if wm is None and cfg.use_working_memory:
        wm = WorkingMemory.load()
    if wm is not None:   # same NACS memory as simulation 1: use its calibrated recall noise
        from .schmeichel2007 import default_config as schmeichel_config
        wm.temperature = schmeichel_config().lstm_temperature
    parts = [run_participant(cfg, "high", seed * 100003 + i, wm) for i in range(cfg.n_high)]
    parts += [run_participant(cfg, "low", seed * 100003 + 5000 + i, wm) for i in range(cfg.n_low)]
    return {"config": cfg, "participants": parts, "stats": analyze(parts)}


def analyze(parts: list[dict]) -> dict:
    out = {}
    for cond in ("high", "low"):
        g = [p for p in parts if p["condition"] == cond]
        e = [p["explicit_measure"] for p in g]
        i = [p["implicit_measure"] for p in g]
        c = [p["chocolates"] for p in g]
        out[cond] = {"r_exp_imp": S.pearson(e, i), "r_exp_choc": S.pearson(e, c),
                     "r_imp_choc": S.pearson(i, c), "chocolates": S.describe(c),
                     "digit_recall": float(np.nanmean([p["digit_recall"] for p in g])),
                     "top_level_share": float(np.mean([p["top_level_share"] for p in g]))}
    out["slopes"] = simple_slopes(parts)
    return out


def simple_slopes(parts: list[dict]) -> dict:
    """
    Moderated regression (as in Friese et al., 2008): chocolates on standardized
    explicit and implicit measures, capacity (high = 1), and their interactions.
    Returns simple slopes per group and predicted chocolates at -1/+1 SD
    (Figure 1).
    """
    e = np.array([p["explicit_measure"] for p in parts]); i = np.array([p["implicit_measure"] for p in parts])
    e = (e - e.mean()) / e.std(ddof=1); i = (i - i.mean()) / i.std(ddof=1)
    y = np.array([p["chocolates"] for p in parts], float)
    out = {}
    for ref in ("high", "low"):
        # code the reference group as 0 so its slopes are the main-effect coefficients
        d = np.array([0.0 if p["condition"] == ref else 1.0 for p in parts])
        X = np.column_stack([np.ones_like(y), e, i, d, e * d, i * d])
        fit = S.ols(y, X, ["const", "explicit", "implicit", "group", "explicit_x_group", "implicit_x_group"])
        b = fit["beta"]
        out[ref] = {
            "explicit": {"b": b["explicit"], "se": fit["se"]["explicit"], "p": fit["p"]["explicit"]},
            "implicit": {"b": b["implicit"], "se": fit["se"]["implicit"], "p": fit["p"]["implicit"]},
            "predicted": {"explicit": [b["const"] - b["explicit"], b["const"] + b["explicit"]],
                          "implicit": [b["const"] - b["implicit"], b["const"] + b["implicit"]]},
        }
        if ref == "high":
            out["interaction_explicit_p"] = fit["p"]["explicit_x_group"]
            out["interaction_implicit_p"] = fit["p"]["implicit_x_group"]
    return out
