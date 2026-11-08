"""
Analysis of M3 (native Soar): the tests of Document 1 for this model.

    python analyze.py                                     # results/grid_seeds_1-200.csv
    python analyze.py --data results/grid_seeds_20001-20200.csv

  H4a  no anticipation effect: the anticipation and constant conditions give identical episodes (structural:
       the conditions differ only in a CLARION drive stimulus, which M3 does not have)
  H4b  no motivational shift: lapse after room A, quiet - free, is below +0.05     (non-inferiority test)
  H4c  no capacity effect with chunking: lapse while busy, busy - free, within +/-0.05  (equivalence test)
  H4d  a capacity effect without chunking (variant N1_no_chunking): busy - free > 0     (paired t-test)
Holm correction over {H4b, H4c, H4d}.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.stats import fci, fmt, fp, h1b_diffs, holm, noninferior, paired_diffs, test_diff, tost  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "results", "grid_seeds_1-200.csv"))
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    L = [f"# M3 native Soar: analysis of {os.path.basename(a.data)}\n"]
    d = df[(df.variant == "native_soar") & (df.scenario == "anticipation")]
    cols = ["lapse_rate", "lapse_approach", "lapse_past", "room_c_cleaned_pct", "steps", "planning_calls"]
    A = d[d.condition == "anticipation"].set_index("seed")[cols].sort_index()
    C = d[d.condition == "constant"].set_index("seed")[cols].sort_index()
    diff = float(np.nan_to_num((A - C).abs().max().max()))
    L.append(f"H4a: largest difference between the anticipation and constant conditions over {len(A)} seeds and "
             f"{len(cols)} measures = {diff} -> {'identical' if diff == 0 else 'NOT identical'}\n")
    tests = [noninferior(paired_diffs(df, "native_soar", "quiet", "free", "lapse_after_A", "shift"), 0.05, "H4b"),
             tost(paired_diffs(df, "native_soar", "busy", "free", "lapse_busy", "capacity"), -0.05, 0.05, "H4c")]
    if "N1_no_chunking" in set(df.variant):
        tests.append(test_diff(paired_diffs(df, "N1_no_chunking", "busy", "free", "lapse_busy", "capacity"), "H4d"))
    holm(tests)
    what = {"H4b": "no shift (quiet - free < +0.05)", "H4c": "no capacity effect, chunking on (within +/-0.05)",
            "H4d": "capacity effect, no chunking (busy - free > 0)"}
    L += ["| Test | What | n seeds | Mean difference [CI] | p (Holm) | Supported |", "|---|---|---|---|---|---|"]
    for r in tests:
        ok = r["p_holm"] < .05 and (r["label"] != "H4d" or r["mean"] > 0)
        L.append(f"| {r['label']} | {what[r['label']]} | {r['n']} | {fmt(r['mean'])} {fci(r.get('ci95') or r.get('ci90'))} | "
                 f"{fp(r['p_holm'])} | {'yes' if ok else 'no'} |")
    L += ["\n## Every variant\n", "| Variant | Mean lapse | Room C % | Steps | Planning calls | Deliberations | Time-outs | H1b | H2 | H3 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v in df.variant.unique():
        dv = df[df.variant == v]
        e = [test_diff(h1b_diffs(df, v), "h"), test_diff(paired_diffs(df, v, "quiet", "free", "lapse_after_A", "shift"), "s"),
             test_diff(paired_diffs(df, v, "busy", "free", "lapse_busy", "capacity"), "c")]
        L.append(f"| {v} | {fmt(dv.lapse_rate.mean())} | {dv.room_c_cleaned_pct.mean():.1f} | {dv.steps.mean():.0f} | "
                 f"{dv.planning_calls.mean():.1f} | {dv.n_deliberations.mean():.1f} | {dv.n_timeouts.mean():.1f} | "
                 + " | ".join(fmt(x["mean"]) for x in e) + " |")
    L += ["\n## Deliberations and time-outs per episode, capacity scenario\n"]
    cap = df[df.scenario == "capacity"].groupby(["variant", "condition"])[["n_deliberations", "n_timeouts", "lapse_busy"]].mean().round(2)
    L.append(cap.to_markdown())
    out = os.path.join(HERE, "results", "analysis_" + os.path.basename(a.data).replace(".csv", ".md"))
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
