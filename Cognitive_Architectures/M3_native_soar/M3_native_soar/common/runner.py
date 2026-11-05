"""
Shared by every package: run many episodes in parallel and write one CSV row per episode.

    run_many(job_fn, jobs, out_csv, workers=2)
job_fn(variant, scenario, condition, seed) must return a dict of measures (one episode).
"""
from __future__ import annotations

import csv
import time
from multiprocessing import Pool

_FN = None


def _init(fn):
    global _FN
    _FN = fn


def _call(job):
    return _FN(*job)


def run_many(job_fn, jobs, out_csv, workers=2, every=500):
    t, rows = time.time(), []
    with Pool(workers, initializer=_init, initargs=(job_fn,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_call, jobs, chunksize=10), 1):
            rows.append(r)
            if i % every == 0:
                print(f"  {i}/{len(jobs)} episodes, {time.time() - t:.0f} s", flush=True)
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    rows.sort(key=lambda r: (r["variant"], r["scenario"], r["condition"], r["seed"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} episodes in {time.time() - t:.0f} s -> {out_csv}")
    return rows
