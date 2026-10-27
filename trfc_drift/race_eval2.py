"""Race evaluation of planner v1 (MPPINumba, protocol-112 settings) or v2 (mppi_race2.MPPIRace2) on named circuits.

Episode protocol as mppi_eval.run (fresh CarSim session and friction probe per episode, one lap from v0 at the start
line), plus the v2 sample-group weights. Circuits: race_A/B/C and dev_race_<9100-9199> (circuits.named), or any
name a protocol registers in circuits.

    .venv-wm\\Scripts\\python.exe -B -m trfc_drift.race_eval2 --planner v2 --layout 4WS_TV --seed 1
        --tracks race_A,dev_race_9100 --mu 0.5,0.8 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trfc_drift import env as E
from trfc_drift.circuits import named
from trfc_drift.evaluate import LOG_COLS

V1 = dict(K=256, T=75, n_sub=5, lam=1.0, w_term=2.0, w_lat_term=0.5, v_margin=0.9, preview_m=150.0, w_spin=20.0,
          beta_safe_deg=40.0, beta_everywhere=True)            # protocol-112 FROZEN planner settings


def run(env, ctrl, track, mu, v0):
    env.reset(track=track, mu=mu, v0_kph=v0, fresh=True)
    ctrl.reset()
    diag, tstep = [], []
    while True:
        t0 = time.perf_counter()
        a = ctrl.act(env)
        tstep.append(time.perf_counter() - t0)
        diag.append(ctrl.last)
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    log = np.array(env.log) if env.log else np.zeros((0, len(LOG_COLS)))
    m = dict(success=bool(info["success"]), reason=info["reason"] or "trunc", t=float(info["t"]),
             s_prog=float(info["s_prog"]), length=float(track.length), laps=int(info["laps"]),
             mean_v=float(np.mean(log[:, 4])) if len(log) else 0.0,
             mean_abs_elat=float(np.mean(np.abs(log[:, 8]))) if len(log) else 0.0,
             max_abs_elat=float(np.max(np.abs(log[:, 8]))) if len(log) else 0.0,
             mu_hat=env.mu_hat, steps=len(tstep), plan_ms_median=float(np.median(tstep) * 1e3),
             plan_ms_p95=float(np.percentile(tstep, 95) * 1e3),
             ess_median=float(np.median([d["ess"] for d in diag])))
    for k in ("w_warm", "w_bias", "w_nom"):
        if k in diag[0]:
            m[k + "_mean"] = float(np.mean([d[k] for d in diag]))
    return m, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=("v1", "v2"), required=True)
    ap.add_argument("--layout", default="4WS_TV")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--mu", default="0.5,0.65,0.8,0.95")
    ap.add_argument("--v0", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--K_b", type=int, default=64)
    ap.add_argument("--K_n", type=int, default=64)
    ap.add_argument("--sig_bias", type=float, default=0.3)
    ap.add_argument("--bias_white", type=float, default=0.5)
    ap.add_argument("--ld_min", type=float, default=8.0)
    ap.add_argument("--ld_t", type=float, default=0.8)
    ap.add_argument("--k_v", type=float, default=2.0)
    ap.add_argument("--v_margin", type=float, default=V1["v_margin"])
    ap.add_argument("--t_max", type=float, default=90.0, help="episode time limit, s (EnvConfig default 90)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = E.EnvConfig(task="race", layout=a.layout, friction_info="estimate", probe=True, record=True, t_max=a.t_max)
    env = E.DDEVEnv(cfg, out / "carsim", seed=a.seed)
    kw = dict(V1, seed=a.seed, world_model="physics2", v_margin=a.v_margin)
    if a.planner == "v1":
        from trfc_drift.mppi_numba import MPPINumba
        ctrl = MPPINumba(cfg, **kw)
    else:
        from trfc_drift.mppi_race2 import MPPIRace2
        ctrl = MPPIRace2(cfg, K_b=a.K_b, K_n=a.K_n, sig_bias=(a.sig_bias,) * 3, bias_white=a.bias_white,
                         ld_min=a.ld_min, ld_t=a.ld_t, k_v=a.k_v, **kw)
    results = []
    try:
        for tname in a.tracks.split(","):
            track = named(tname, w=cfg.width)
            for mu in [float(m) for m in a.mu.split(",")]:
                t0 = time.perf_counter()
                m, log = run(env, ctrl, track, mu, a.v0)
                m.update(track=tname, mu=mu, v0=a.v0, wall_s=time.perf_counter() - t0)
                results.append(m)
                np.save(out / f"traj_{tname}_mu{mu:.2f}.npy", log)
                (out / "race_eval2_partial.json").write_text(json.dumps(dict(args=vars(a), results=results), indent=1))
                print(json.dumps(m), flush=True)
    finally:
        env.close()
    (out / "race_eval2.json").write_text(json.dumps(dict(args=vars(a), results=results), indent=1))


if __name__ == "__main__":
    main()
