"""
Run ONE episode of one variant and show what happened.

    python run_episode.py --variant clarion --scenario anticipation --condition anticipation --seed 1
    python run_episode.py ... --trace            # print every decision (what the model saw and chose)
    python run_episode.py ... --trace-csv t.csv  # save the decisions to a CSV file

Variants, scenarios and conditions: python run_episode.py --list
"""
import argparse
import csv
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import package  # noqa: E402

model = importlib.import_module(package.MODEL)
from common.measures import all_conditions  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=f"Run one episode: {package.TITLE}")
    ap.add_argument("--variant", default=next(iter(model.VARIANTS)))
    ap.add_argument("--scenario", default="anticipation")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--seed", type=int, default=5001, help="5001+ are development seeds; 1-200 were the confirmatory seeds")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-csv")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("variants:  ", ", ".join(model.VARIANTS))
        print("conditions:", ", ".join(f"{s}/{c}" for s, c in all_conditions()))
        return
    cond = a.condition or dict((s, c) for s, c in reversed(all_conditions()))[a.scenario]
    m, ep = model.run_one(a.variant, a.scenario, cond, a.seed, return_episode=True)
    rows = [model.trace_row(d) for d in ep["decisions"]]
    if a.trace:
        keys = list(rows[0].keys())
        for r in rows:
            print("  ".join(f"{k}={r.get(k)}" for k in keys if k in r))
    if a.trace_csv:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(a.trace_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} decisions written to {a.trace_csv}")
    print(f"\n{package.TITLE}: {a.variant}, {a.scenario}/{cond}, seed {a.seed}")
    for k in ("steps", "finished", "lapse_rate", "lapse_approach", "lapse_past", "lapse_busy", "room_c_cleaned_pct",
              "planning_calls", "quiet_violations", "n_exposed", "n_deliberations", "n_timeouts"):
        if k in m:
            print(f"  {k:<20} {m[k]}")


if __name__ == "__main__":
    main()
