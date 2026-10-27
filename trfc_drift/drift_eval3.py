"""Drift evaluation of planner v1 (MPPINumba) or v3 (mppi_race3.MPPIRace3: mixture proposals + axle-aware braking
envelope) with the protocol-111 drift settings, on named drift courses (evaluate.get_track: drift_A/B, dev_<seed>).

The episode loop is mppi_eval.run (frozen by protocol 111, imported unchanged): fresh CarSim session and friction
probe per episode, drift-zone scoring. Adds track length and the drift score D = completion x zone accuracy.

    .venv-wm\\Scripts\\python.exe -B -m trfc_drift.drift_eval3 --planner v3 --layout 2WS_RWD --seed 1
        --tracks dev_9011,dev_9012 --mu 0.35,0.55,0.75,0.95 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trfc_drift import env as E
from trfc_drift.evaluate import get_track
from trfc_drift.mppi_eval import run

DRIFT111 = dict(K=256, T=75, n_sub=5, lam=1.0, w_term=2.0, w_lat_term=0.5, v_margin=0.9, preview_m=50.0, w_spin=20.0,
                beta_safe_deg=40.0, beta_everywhere=True)      # protocol-111 FROZEN planner settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=("v1", "v3"), required=True)
    ap.add_argument("--layout", default="4WS_TV")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--mu", default="0.35,0.55,0.75,0.95")
    ap.add_argument("--v0", type=float, default=65.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = E.EnvConfig(task="drift", layout=a.layout, friction_info="estimate", probe=True, record=True)
    env = E.DDEVEnv(cfg, out / "carsim", seed=a.seed)
    kw = dict(DRIFT111, seed=a.seed, world_model="physics2")
    if a.planner == "v1":
        from trfc_drift.mppi_numba import MPPINumba as Planner
    else:
        from trfc_drift.mppi_race3 import MPPIRace3 as Planner
    ctrl = Planner(cfg, **kw)
    results = []
    try:
        for tname in a.tracks.split(","):
            track = get_track(tname, cfg.width)
            for mu in [float(m) for m in a.mu.split(",")]:
                t0 = time.perf_counter()
                m, log = run(env, ctrl, track, mu, a.v0)
                comp = min(m["s_prog"] / track.length, 1.0)
                m.update(track=tname, mu=mu, v0=a.v0, length=float(track.length), completion=comp,
                         D=comp * (m["zone_frac"] or 0.0), wall_s=time.perf_counter() - t0)
                results.append(m)
                np.save(out / f"traj_{tname}_mu{mu:.2f}.npy", log)
                (out / "drift_eval3_partial.json").write_text(json.dumps(dict(args=vars(a), results=results), indent=1))
                print(json.dumps(m), flush=True)
    finally:
        env.close()
    (out / "drift_eval3.json").write_text(json.dumps(dict(args=vars(a), results=results), indent=1))


if __name__ == "__main__":
    main()
