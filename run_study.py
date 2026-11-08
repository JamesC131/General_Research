"""
Run every variant of this model in all six conditions, for many seeds.

    python run_study.py                         # seeds 1-200, all variants -> results/my_run.csv
    python run_study.py --seeds 20 --variants clarion,A1_fixed_drives --out results/quick.csv
Then:  python analyze.py --data results/my_run.csv
"""
import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import package  # noqa: E402

model = importlib.import_module(package.MODEL)
from common.measures import all_conditions  # noqa: E402
from common.runner import run_many  # noqa: E402


def job(variant, scenario, condition, seed):
    return model.run_one(variant, scenario, condition, seed)


def main():
    ap = argparse.ArgumentParser(description=f"Run the study for {package.TITLE}")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--variants", default=",".join(model.VARIANTS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "my_run.csv"))
    a = ap.parse_args()
    jobs = [(v, s, c, seed) for v in a.variants.split(",") for s, c in all_conditions()
            for seed in range(a.seed0, a.seed0 + a.seeds)]
    print(f"{package.TITLE}: {len(jobs)} episodes")
    run_many(job, jobs, a.out, workers=a.workers)


if __name__ == "__main__":
    main()
