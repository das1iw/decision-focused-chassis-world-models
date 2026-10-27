"""Evaluate the MPPI world-model controller on fixed tracks and friction values (same episode protocol as evaluate.py).

    .venv-wm\\Scripts\\python.exe -B -m trfc_drift.mppi_eval --tracks dev_9001 --mu 0.7 --v0 65 --out <dir>
        [--layout 4WS_TV] [--K 256] [--T 40] [--n_sub 5] [--lam 2] [--device cpu] [--max_steps 0]

Per (track, mu): one episode, fresh CarSim session, protocol-106 probe first (mu_hat), then MPPI every 20 ms.
Metrics as evaluate.py plus planner diagnostics (time per step, effective sample size, survivor fraction).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trfc_drift import env as E
from trfc_drift.evaluate import LOG_COLS, get_track
from trfc_drift.mppi import MPPI


def run(env, ctrl, track, mu, v0, max_steps=0):
    env.reset(track=track, mu=mu, v0_kph=v0, fresh=True)
    ctrl.reset()
    zone = zone_ok = 0
    zb, diag, tstep = [], [], []
    n = 0
    while True:
        t0 = time.perf_counter()
        a = ctrl.act(env)
        tstep.append(time.perf_counter() - t0)
        diag.append(ctrl.last)
        _, r, term, trunc, info = env.step(a)
        n += 1
        if info["in_zone"] > 0:
            zone += 1
            zone_ok += abs(info["beta"] - info["beta_ref"]) <= np.radians(5.0)
            zb.append(abs(info["beta"]))
        if term or trunc or (max_steps and n >= max_steps):
            break
    log = np.array(env.log) if env.log else np.zeros((0, len(LOG_COLS)))
    return dict(success=bool(info["success"]), reason=info["reason"] or "trunc", t=float(info["t"]),
                s_prog=float(info["s_prog"]), mean_v=float(np.mean(log[:, 4])) if len(log) else 0.0,
                mean_abs_elat=float(np.mean(np.abs(log[:, 8]))) if len(log) else 0.0,
                max_abs_elat=float(np.max(np.abs(log[:, 8]))) if len(log) else 0.0,
                zone_steps=int(zone), zone_frac=float(zone_ok / zone) if zone else None,
                zone_mean_abs_beta_deg=float(np.degrees(np.mean(zb))) if zb else None,
                mu_hat=env.mu_hat, laps=int(info["laps"]), steps=n,
                plan_ms_median=float(np.median(tstep) * 1e3), ess_median=float(np.median([d["ess"] for d in diag])),
                alive_median=float(np.median([d["alive"] for d in diag])),
                w_policy_mean=float(np.mean([d.get("w_policy", 0.0) for d in diag]))), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="dev_9001")
    ap.add_argument("--mu", default="0.7")
    ap.add_argument("--v0", type=float, default=65.0)
    ap.add_argument("--layout", default="4WS_TV")
    ap.add_argument("--task", default="drift")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--n_sub", type=int, default=5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--w_term", type=float, default=0.05)
    ap.add_argument("--w_lat_term", type=float, default=0.0)
    ap.add_argument("--v_margin", type=float, default=1.0)
    ap.add_argument("--preview_m", type=float, default=50.0, help="envelope terminal-cost preview beyond the horizon")
    ap.add_argument("--w_spin", type=float, default=0.0)
    ap.add_argument("--beta_safe_deg", type=float, default=40.0)
    ap.add_argument("--beta_everywhere", action="store_true")
    ap.add_argument("--sac_ckpt", default=None, help="final.pt of a SAC run (same layout and info): MPPI + SAC")
    ap.add_argument("--K_pi", type=int, default=64)
    ap.add_argument("--w_v", type=float, default=1.0)
    ap.add_argument("--pi_noise", type=float, default=1.0)
    ap.add_argument("--info", default="estimate", help="env friction_info (must match the SAC run for --sac_ckpt)")
    ap.add_argument("--world_model", default="physics2", help="physics2 | hybrid:<checkpoint .pt>")
    ap.add_argument("--numba", action="store_true", help="compiled planner (mppi_numba.MPPINumba, physics2 only)")
    ap.add_argument("--gpu_graph", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = E.EnvConfig(task=a.task, layout=a.layout, friction_info=a.info, probe=True, record=True)
    env = E.DDEVEnv(cfg, out / "carsim", seed=a.seed)
    kw = dict(K=a.K, T=a.T, n_sub=a.n_sub, lam=a.lam, seed=a.seed, w_term=a.w_term, w_lat_term=a.w_lat_term,
              v_margin=a.v_margin, w_spin=a.w_spin, beta_safe_deg=a.beta_safe_deg,
              beta_everywhere=a.beta_everywhere, world_model=a.world_model, preview_m=a.preview_m)
    if a.numba:
        import numba
        numba.set_num_threads(a.threads)
        from trfc_drift.mppi_numba import MPPINumba
        ctrl = MPPINumba(cfg, **kw)
    elif a.sac_ckpt:
        from trfc_drift.mppi_sac import ValueMPPI
        ctrl = ValueMPPI(cfg, a.sac_ckpt, device=a.device, K_pi=a.K_pi, w_v=a.w_v, pi_noise=a.pi_noise, **kw)
    elif a.gpu_graph:
        from trfc_drift.mppi_gpu import MPPIGraph
        ctrl = MPPIGraph(cfg, device="cuda", **kw)
    else:
        ctrl = MPPI(cfg, device=a.device, **kw)
    results, trajs = [], {}
    try:
        for tname in a.tracks.split(","):
            track = get_track(tname, cfg.width)
            for mu in [float(m) for m in a.mu.split(",")]:
                t0 = time.perf_counter()
                m, log = run(env, ctrl, track, mu, a.v0, a.max_steps)
                m.update(track=tname, mu=mu, v0=a.v0, wall_s=time.perf_counter() - t0)
                results.append(m)
                trajs[f"{tname}_mu{mu:.2f}"] = log
                np.save(out / f"traj_{tname}_mu{mu:.2f}.npy", log)      # per episode, written immediately
                (out / "mppi_eval_partial.json").write_text(json.dumps(dict(args=vars(a), results=results), indent=1))
                print(json.dumps(m), flush=True)
    finally:
        env.close()
    (out / "mppi_eval.json").write_text(json.dumps(dict(args=vars(a), results=results), indent=1))
    np.savez_compressed(out / "mppi_trajectories.npz", **trajs)


if __name__ == "__main__":
    main()
