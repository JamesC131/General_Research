"""
Check that this package reproduces the stored results exactly.

For every variant in a stored results file, re-runs the first K seeds in all six conditions and compares
every measure with the stored row. Any difference is printed.

    python verify.py                              # default: results/grid_seeds_1-200.csv, K = 3
    python verify.py --data results/grid_seeds_20001-20200.csv --k 2
"""
import argparse
import importlib
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import package  # noqa: E402

model = importlib.import_module(package.MODEL)


def same(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "results", "grid_seeds_1-200.csv"))
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    checked = bad = 0
    for v in sorted(set(df.variant) & set(model.VARIANTS)):
        d = df[df.variant == v]
        for seed in sorted(d.seed.unique())[:a.k]:
            for _, row in d[d.seed == seed].iterrows():
                new = model.run_one(v, row.scenario, row.condition, int(seed))
                diffs = [k for k in new if k in row.index and not same(new[k], row[k])]
                checked += 1
                if diffs:
                    bad += 1
                    print(f"DIFFERENT: {v} {row.scenario}/{row.condition} seed {seed}: {diffs}")
    print(f"{package.TITLE}: {checked} episodes re-run, {checked - bad} identical to {os.path.basename(a.data)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
