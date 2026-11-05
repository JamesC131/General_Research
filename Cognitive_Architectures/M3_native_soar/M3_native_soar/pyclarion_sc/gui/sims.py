"""
The four simulations, packaged for the Simulation Lab.

Each entry knows how to: list its parameters (with their source tag), run
once, turn the statistics into comparison rows and chart data, describe its
participants, and reduce a run to flat numbers for replications.
"""
from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields

import numpy as np

from ..core.nacs import WorkingMemory
from ..experiments import friese2008 as FR
from ..experiments import myrseth2009 as MY
from ..experiments import schmeichel2007 as SC

MAX_N = 500
MAX_REPS = 200
HIDDEN = {"interpretation", "hidden", "epochs", "choice_hidden", "attitude_hidden", "use_working_memory",
          "train_networks", "video_steps"}

_WM = None


def _wm():
    global _WM
    if _WM is None:
        _WM = WorkingMemory.load()
    return _WM


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def _config(sim_id: str, overrides: dict):
    spec = SIMS[sim_id]
    base = spec["default"]()
    clean = {}
    for f in fields(base):
        if f.name not in overrides or f.name in HIDDEN:
            continue
        v, cur = overrides[f.name], getattr(base, f.name)
        if isinstance(cur, bool):
            v = bool(v)
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        elif isinstance(cur, tuple):
            v = tuple(type(c)(x) for c, x in zip(cur, v))
        clean[f.name] = v
    for k in ("n_before", "n_after", "n_high", "n_low"):
        if k in clean:
            clean[k] = max(2, min(clean[k], MAX_N))
    for k in ("n_test1", "n_test2"):
        if k in clean:
            clean[k] = tuple(max(2, min(x, MAX_N)) for x in clean[k])
    return spec["default"](**clean)


def params(sim_id: str) -> list[dict]:
    cfg = SIMS[sim_id]["default"]()
    out = []
    for f in fields(cfg):
        if f.name in HIDDEN:
            continue
        v = getattr(cfg, f.name)
        kind = "bool" if isinstance(v, bool) else "int" if isinstance(v, int) else \
            "float" if isinstance(v, float) else "pair" if isinstance(v, tuple) else "text"
        out.append({"name": f.name, "label": f.name.replace("_", " ").capitalize(),
                    "value": list(v) if isinstance(v, tuple) else v, "kind": kind,
                    "source": f.metadata.get("source", "chosen"), "note": f.metadata.get("note", "")})
    return out


# ---------------------------------------------------------------------------
# Myrseth
# ---------------------------------------------------------------------------

def _myr_run(interp):
    def run(overrides, seed):
        return MY.run(_config(f"myrseth_{'ei' if interp == 'explicit_implicit' else 'dg'}", overrides), seed=seed)
    return run


def _myr_flat(stats):
    f = {}
    for c in ("before", "after"):
        for food in ("granola", "chocolate"):
            f[f"{c}_{food}"] = stats[c][food]["M"]
        f[f"{c}_t"], f[f"{c}_p"] = stats[c]["test"]["t"], stats[c]["test"]["p"]
        f[f"{c}_explicitness"] = stats[c]["mean_explicitness"]
    f["F"], f["F_p"] = stats["interaction"]["F"], stats["interaction"]["p"]
    return f


def _myr_pattern(f):
    return bool(f["before_p"] < .05 and f["before_t"] > 0 and f["after_p"] >= .05)


def _myr_view(interp):
    pub = MY.PUBLISHED_SIM[interp]

    def view(res):
        s = res["stats"]
        rows = []
        for c, cn in (("before", "Before choice"), ("after", "After choice")):
            for food in ("granola", "chocolate"):
                h = MY.HUMAN[c][food]
                rows.append({"group": cn, "label": f"{food.capitalize()} rating, M (SD)",
                             "human": f"{h[0]:.1f} ({h[1]:.1f})", "published": f"{pub[c][food][0]} ({pub[c][food][1]})",
                             "model": f"{s[c][food]['M']:.1f} ({s[c][food]['SD']:.1f})"})
            t = s[c]["test"]
            rows.append({"group": cn, "label": "Granola vs chocolate, paired t (d)",
                         "human": f"{MY.HUMAN[c]['t']} ({MY.HUMAN[c]['d']})", "published": f"{pub[c]['t']} ({pub[c]['d']})",
                         "model": f"{t['t']:.2f} ({t['d']:.2f}), {_p(t['p'])}"})
        it = s["interaction"]
        rows.append({"group": "Both groups", "label": "Choice x food interaction, F(1, 32)",
                     "human": "significant", "published": f"{pub['F']}", "model": f"{it['F']:.2f}, {_p(it['p'])}"})
        panels = [{"title": cn, "ylabel": "Appeal rating (0-150)", "categories": ["Granola bars", "Chocolate bars"],
                   "model": [s[c]["granola"]["M"], s[c]["chocolate"]["M"]],
                   "model_sd": [s[c]["granola"]["SD"], s[c]["chocolate"]["SD"]],
                   "human": [MY.HUMAN[c]["granola"][0], MY.HUMAN[c]["chocolate"][0]],
                   "published": [pub[c]["granola"][0], pub[c]["chocolate"][0]], "ymin": 0, "ymax": 150}
                  for c, cn in (("before", "Before choice"), ("after", "After choice"))]
        mech = {"title": "Explicitness (probability the top level acts)",
                "categories": ["Before choice", "After choice"],
                "values": [s["before"]["mean_explicitness"], s["after"]["mean_explicitness"]]}
        parts = [{"id": i + 1, "condition": p["condition"], "granola": round(p["granola"], 1),
                  "chocolate": round(p["chocolate"], 1), "choice": p["choice"].replace("select_", ""),
                  "goal": p["goal"].replace("_", " "), "explicitness": round(p["mean_explicitness"], 2),
                  "top_share": round(p["top_level_share"], 2), "events": p["events"]}
                 for i, p in enumerate(res["participants"])]
        return {"rows": rows, "panels": panels, "mechanism": mech, "participants": parts, "detail": "events",
                "columns": ["id", "condition", "granola", "chocolate", "choice", "goal", "explicitness", "top_share"]}
    return view


# ---------------------------------------------------------------------------
# Friese
# ---------------------------------------------------------------------------

def _fr_run(overrides, seed):
    return FR.run(_config("friese2008", overrides), seed=seed, wm=_wm())


def _fr_flat(stats):
    f = {}
    for c in ("high", "low"):
        for m in ("r_exp_imp", "r_exp_choc", "r_imp_choc"):
            f[f"{c}_{m}"] = stats[c][m]["r"]
        for m in ("explicit", "implicit"):
            f[f"{c}_slope_{m}"], f[f"{c}_slope_{m}_p"] = stats["slopes"][c][m]["b"], stats["slopes"][c][m]["p"]
        f[f"{c}_chocolates"] = stats[c]["chocolates"]["M"]
    return f


def _fr_pattern(f):
    return bool(f["high_r_exp_choc"] > f["high_r_imp_choc"] and f["low_r_imp_choc"] > f["low_r_exp_choc"])


def _fr_view(res):
    s = res["stats"]
    rows = []
    names = {"r_exp_imp": "Explicit x implicit", "r_exp_choc": "Explicit x chocolates", "r_imp_choc": "Implicit x chocolates"}
    for c, cn in (("high", "High capacity"), ("low", "Low capacity")):
        for m, mn in names.items():
            rows.append({"group": cn, "label": f"{mn}, r", "human": f"{FR.HUMAN[c][m]:.2f}",
                         "published": f"{FR.PUBLISHED_SIM[c][m]:.2f}", "model": f"{s[c][m]['r']:.2f}, {_p(s[c][m]['p'])}"})
        for m in ("explicit", "implicit"):
            sl = s["slopes"][c][m]
            rows.append({"group": cn, "label": f"Simple slope, {m} measure", "human":
                         "predicts" if (c, m) in (("high", "explicit"), ("low", "implicit")) else "weaker",
                         "published": "same as human", "model": f"b = {sl['b']:.2f}, {_p(sl['p'])}"})
    panels = [{"title": cn, "ylabel": "Correlation (r)", "categories": list(names.values()),
               "model": [s[c][m]["r"] for m in names], "human": [FR.HUMAN[c][m] for m in names],
               "published": [FR.PUBLISHED_SIM[c][m] for m in names], "ymin": -0.4, "ymax": 1.0}
              for c, cn in (("high", "High capacity"), ("low", "Low capacity"))]
    mech = {"title": "Share of picks made by the top level", "categories": ["High capacity", "Low capacity"],
            "values": [s["high"]["top_level_share"], s["low"]["top_level_share"]]}
    slopes = {c: s["slopes"][c]["predicted"] for c in ("high", "low")}
    parts = [{"id": i + 1, "condition": p["condition"], "explicitness": round(p["explicitness"], 2),
              "explicit_measure": round(p["explicit_measure"], 2), "implicit_measure": round(p["implicit_measure"], 2),
              "chocolates": p["chocolates"], "top_share": round(p["top_level_share"], 2),
              "picks": [{"item": it, "level": lv} for it, lv in zip(p["picks"], p["levels"])]}
             for i, p in enumerate(res["participants"])]
    return {"rows": rows, "panels": panels, "mechanism": mech, "slopes": slopes, "participants": parts,
            "detail": "picks", "columns": ["id", "condition", "explicitness", "explicit_measure", "implicit_measure",
                                           "chocolates", "top_share"]}


# ---------------------------------------------------------------------------
# Schmeichel
# ---------------------------------------------------------------------------

def _sc_run(overrides, seed):
    return SC.run(_config("schmeichel2007", overrides), seed=seed, wm=_wm())


def _sc_flat(stats):
    f = {}
    for t in ("test1", "test2"):
        for g in ("attention_control", "no_control"):
            f[f"{t}_{g}"] = stats[t][g]["M"]
        f[f"{t}_t"], f[f"{t}_p"] = stats[t]["test"]["t"], stats[t]["test"]["p"]
    for g in ("attention_control", "no_control"):
        f[f"{g}_explicitness_memory"] = stats[g]["explicitness_activity2"]
    return f


def _sc_pattern(f):
    return bool(f["test1_no_control"] > f["test1_attention_control"] and f["test2_no_control"] > f["test2_attention_control"])


def _sc_view(res):
    s = res["stats"]
    rows = []
    groups = (("attention_control", "Attention control"), ("no_control", "No control"))
    for t, tn in (("test1", "Test 1: operation span"), ("test2", "Test 2: sentence span")):
        for g, gn in groups:
            rows.append({"group": tn, "label": f"{gn}, items recalled M (SD)", "human": f"{SC.HUMAN[t][g]:.2f}",
                         "published": f"{SC.PUBLISHED_SIM[t][g]}", "model": f"{s[t][g]['M']:.1f} ({s[t][g]['SD']:.1f})"})
        tt = s[t]["test"]
        rows.append({"group": tn, "label": "Group difference", "human": "not significant",
                     "published": "not significant", "model": f"t = {tt['t']:.2f}, {_p(tt['p'])}"})
    panels = [{"title": tn, "ylabel": "Items recalled", "categories": [gn for _, gn in groups],
               "model": [s[t][g]["M"] for g, _ in groups], "model_sd": [s[t][g]["SD"] for g, _ in groups],
               "human": [SC.HUMAN[t][g] for g, _ in groups], "published": [SC.PUBLISHED_SIM[t][g] for g, _ in groups],
               "ymin": 0, "ymax": 50}
              for t, tn in (("test1", "Test 1: operation span"), ("test2", "Test 2: sentence span"))]
    mech = {"title": "Explicitness during the memory test", "categories": [gn for _, gn in groups],
            "values": [s[g]["explicitness_activity2"] for g, _ in groups]}
    parts = [{"id": i + 1, "condition": p["group"].replace("_", " "), "test": p["test"].replace("test", "Test "),
              "score": p["score"], "start_deficit": round(p["timeline"][0]["deficit"] + 0, 2),
              "final_deficit": round(p["final_deficit"], 2),
              "explicitness_video_end": round(p["explicitness_activity1_end"], 2),
              "explicitness_memory": round(p["explicitness_activity2"], 2),
              "control_failures": None if np.isnan(p["control_failures"]) else round(p["control_failures"], 2),
              "timeline": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()} for e in p["timeline"]]}
             for i, p in enumerate(res["participants"])]
    return {"rows": rows, "panels": panels, "mechanism": mech, "participants": parts, "detail": "timeline",
            "columns": ["id", "condition", "test", "score", "final_deficit", "explicitness_video_end",
                        "explicitness_memory", "control_failures"]}


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _p(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


SIMS = {
    "schmeichel2007": {
        "title": "Self-control and memory", "study": "Schmeichel (2007)", "paper": "Gomez & Sun (2024)",
        "summary": "After ignoring distractor words, people did slightly worse on two working-memory tests. "
                   "In the model, exerting control satisfies the achievement drive, its deficit falls (Eq. 6), "
                   "and explicit control of memory weakens.",
        "default": SC.default_config, "run": _sc_run, "flat": _sc_flat, "pattern": _sc_pattern, "view": _sc_view,
        "pattern_text": "No-control group recalls more on both tests",
        "key": ("test2_t", "Test 2 group difference (t)"), "key_human": None, "seconds": 3,
    },
    "friese2008": {
        "title": "Explicit and implicit attitudes in food choice", "study": "Friese, Hofmann & Waenke (2008)",
        "paper": "Gomez & Sun (2024)",
        "summary": "With full cognitive capacity, explicit attitudes predicted how many chocolates people chose; "
                   "under memory load, implicit attitudes did. In the model, capacity sets explicitness directly "
                   "(0.7-1.0 vs 0.4-0.6).",
        "default": FR.default_config, "run": _fr_run, "flat": _fr_flat, "pattern": _fr_pattern, "view": _fr_view,
        "pattern_text": "Explicit predicts under high capacity, implicit under low",
        "key": ("low_r_imp_choc", "Low capacity: implicit x chocolates (r)"), "key_human": 0.45, "seconds": 5,
    },
    "myrseth_ei": {
        "title": "Valuation before and after choice: explicit-implicit", "study": "Myrseth, Fishbach & Trope (2009)",
        "paper": "Gomez & Sun (2025), interpretation 1",
        "summary": "People rated granola above chocolate when a choice was still ahead, but not after choosing. "
                   "In the model, the pending choice stimulates the achievement drive, which raises the utility "
                   "of explicit (healthy) rules.",
        "default": lambda **kw: MY.default_config("explicit_implicit", **kw), "run": _myr_run("explicit_implicit"),
        "flat": _myr_flat, "pattern": _myr_pattern, "view": _myr_view("explicit_implicit"),
        "pattern_text": "Granola > chocolate before choice (p < .05), no difference after",
        "key": ("before_t", "Before-choice difference (t)"), "key_human": 2.82, "seconds": 2,
    },
    "myrseth_dg": {
        "title": "Valuation before and after choice: desire-goal", "study": "Myrseth, Fishbach & Trope (2009)",
        "paper": "Gomez & Sun (2025), interpretation 2",
        "summary": "The same data read as a competition between a desire for tasty food and the goal of eating "
                   "healthily. Top-level rules create a 'choose healthy food' subgoal that informs both levels.",
        "default": lambda **kw: MY.default_config("desire_goal", **kw), "run": _myr_run("desire_goal"),
        "flat": _myr_flat, "pattern": _myr_pattern, "view": _myr_view("desire_goal"),
        "pattern_text": "Granola > chocolate before choice (p < .05), no difference after",
        "key": ("before_t", "Before-choice difference (t)"), "key_human": 2.82, "seconds": 3,
    },
}


def catalog() -> list[dict]:
    return [{"id": k, **{f: v[f] for f in ("title", "study", "paper", "summary", "pattern_text", "seconds", "key_human")},
             "key_label": v["key"][1],
             "params": params(k)} for k, v in SIMS.items()]


def run_once(sim_id: str, overrides: dict, seed: int) -> dict:
    spec = SIMS[sim_id]
    res = spec["run"](overrides, int(seed))
    view = spec["view"](res)
    flat = spec["flat"](res["stats"])
    view["pattern"] = {"text": spec["pattern_text"], "achieved": spec["pattern"](flat)}
    view["seed"] = int(seed)
    return view


# ---------------------------------------------------------------------------
# replication jobs (run in worker processes)
# ---------------------------------------------------------------------------

def _rep(sim_id: str, overrides: dict, seed: int) -> dict:
    spec = SIMS[sim_id]
    flat = spec["flat"](spec["run"](overrides, seed)["stats"])
    flat["_pattern"] = spec["pattern"](flat)
    return flat


_POOL = None
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2)))
    return _POOL


def start_replications(sim_id: str, overrides: dict, reps: int) -> str:
    reps = max(2, min(int(reps), MAX_REPS))
    _config(sim_id, overrides)  # validate early
    job_id = uuid.uuid4().hex[:10]
    job = {"id": job_id, "sim": sim_id, "total": reps, "done": 0, "rows": [], "error": None, "cancelled": False}
    with _LOCK:
        _JOBS[job_id] = job
        for k in [k for k, j in _JOBS.items() if j["done"] >= j["total"]][:-20]:   # keep memory small
            _JOBS.pop(k, None)
    futures = [_pool().submit(_rep, sim_id, overrides, s) for s in range(1, reps + 1)]

    def collect(fut):
        with _LOCK:
            if fut.cancelled():
                job["done"] += 1
                return
            exc = fut.exception()
            if exc is not None:
                job["error"] = str(exc)
            else:
                job["rows"].append(fut.result())
            job["done"] += 1

    for f in futures:
        f.add_done_callback(collect)
    job["futures"] = futures
    return job_id


def cancel(job_id: str) -> None:
    job = _JOBS.get(job_id)
    if job:
        job["cancelled"] = True
        for f in job.get("futures", []):
            f.cancel()


def job_status(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    with _LOCK:
        rows = list(job["rows"])
    spec = SIMS[job["sim"]]
    out = {"id": job_id, "total": job["total"], "done": job["done"], "error": job["error"],
           "finished": job["done"] >= job["total"], "n": len(rows)}
    if rows:
        keys = [k for k in rows[0] if not k.startswith("_") and not k.endswith("_p")]
        out["pattern_rate"] = float(np.mean([r["_pattern"] for r in rows]))
        out["metrics"] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                              "lo": float(np.percentile([r[k] for r in rows], 2.5)),
                              "hi": float(np.percentile([r[k] for r in rows], 97.5))} for k in keys}
        key, label = spec["key"]
        out["key"] = {"name": key, "label": label, "values": [r[key] for r in rows]}
        out["rows"] = rows
    return out
