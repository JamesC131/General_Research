"""
pyclarion-sc web app: Core Studio (/core), Simulation Lab (/lab) and Robot World (/robot).

Run:  pyclarion-sc-gui --host 127.0.0.1 --port 8050
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..robot import catalog as robot_catalog
from ..robot import controller as robot
from . import core_engine, sims

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="pyclarion-sc", version=__version__, docs_url="/api/docs", redoc_url=None)


def _clean(o):
    """Make results JSON-safe (NumPy types, NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


class CoreRun(BaseModel):
    scenario: dict
    steps: int = Field(40, ge=1, le=core_engine.LIMITS["steps"])
    seed: int = 1


class RobotRun(BaseModel):
    scenario: str = "anticipation"
    condition: str | None = None
    controller: str = "clarion"
    seed: int = Field(1, ge=0, le=10_000_000)


class SimRun(BaseModel):
    params: dict = {}
    seed: int = 1


class SimReplicate(BaseModel):
    params: dict = {}
    reps: int = Field(50, ge=2, le=sims.MAX_REPS)


# ---- pages ----------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/core")


@app.get("/core", include_in_schema=False)
def core_page():
    return FileResponse(STATIC / "core.html")


@app.get("/lab", include_in_schema=False)
def lab_page():
    return FileResponse(STATIC / "lab.html")


@app.get("/guide", include_in_schema=False)
def guide_page():
    return FileResponse(STATIC / "guide.html")


@app.get("/robot", include_in_schema=False)
def robot_page():
    return FileResponse(STATIC / "robot.html")


# ---- Core Studio API ------------------------------------------------------------

@app.get("/api/core/presets")
def presets():
    return core_engine.PRESETS


@app.post("/api/core/run")
def core_run(body: CoreRun):
    try:
        return _clean(core_engine.run(body.scenario, body.steps, body.seed))
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, str(e))


# ---- Robot World API ----------------------------------------------------------------

@app.get("/api/robot/scenarios")
def robot_scenarios():
    return [{"id": k, "title": v["title"], "paper": v["paper"], "story": v["story"],
             "conditions": [{"id": c, "label": lab} for c, (lab, _) in v["conditions"].items()]}
            for k, v in robot.SCENARIOS.items()]


@app.get("/api/robot/index")
def robot_index():
    return _clean(robot_catalog.index())


@app.post("/api/robot/run")
def robot_run(body: RobotRun):
    if body.scenario not in robot.SCENARIOS:
        raise HTTPException(404, f"Unknown scenario '{body.scenario}'.")
    conds = robot.SCENARIOS[body.scenario]["conditions"]
    if body.condition is not None and body.condition not in conds:
        raise HTTPException(400, f"Unknown condition '{body.condition}' for this scenario.")
    if body.controller not in robot.CONTROLLERS:
        raise HTTPException(400, f"Unknown robot '{body.controller}'.")
    return _clean(robot.run_episode(body.scenario, body.condition, seed=body.seed, controller=body.controller))


# ---- Simulation Lab API ----------------------------------------------------------

@app.get("/api/sims")
def sim_catalog():
    return _clean(sims.catalog())


@app.post("/api/sims/{sim_id}/run")
def sim_run(sim_id: str, body: SimRun):
    if sim_id not in sims.SIMS:
        raise HTTPException(404, f"Unknown simulation '{sim_id}'.")
    try:
        return _clean(sims.run_once(sim_id, body.params, body.seed))
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/sims/{sim_id}/replicate")
def sim_replicate(sim_id: str, body: SimReplicate):
    if sim_id not in sims.SIMS:
        raise HTTPException(404, f"Unknown simulation '{sim_id}'.")
    try:
        return {"job": sims.start_replications(sim_id, body.params, body.reps)}
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    try:
        return _clean(sims.job_status(job_id))
    except KeyError:
        raise HTTPException(404, "This replication job no longer exists. Start it again.")


@app.delete("/api/jobs/{job_id}")
def job_cancel(job_id: str):
    sims.cancel(job_id)
    return {"cancelled": job_id}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    ap = argparse.ArgumentParser(description="pyclarion-sc web app (Core Studio and Simulation Lab)")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    import uvicorn
    print(f"Core Studio:     http://{a.host}:{a.port}/core\nSimulation Lab:  http://{a.host}:{a.port}/lab\nRobot World:     http://{a.host}:{a.port}/robot")
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
