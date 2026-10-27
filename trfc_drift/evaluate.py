"""Deterministic evaluation of a trained DDEV policy on fixed tracks and friction values.

    .venv-wm\\Scripts\\python.exe -B -m trfc_drift.evaluate --run D:\\TRFC_data\\drift_runs\\<name>
        [--tracks drift_A,drift_B] [--mu 0.4,0.7,0.95] [--v0 60] [--tag eval]

Per (track, mu): one episode with the policy mean action. Metrics:
- success and termination reason; time; progress; mean speed;
- mean and max |e_lat|;
- drift zones: fraction of zone steps with |beta - beta_ref| <= 5 deg, mean |beta| in zones;
- lap time (race, completed laps only).
Trajectories are saved to <run>/<tag>_trajectories.npz and the metrics to <run>/<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from trfc_drift import env as E
from trfc_drift import tracks as T
from trfc_drift.sac import Actor

LOG_COLS = ["t", "x", "y", "yaw", "vx", "vy", "r", "beta", "e_lat", "s_prog", "steer_FL", "steer_FR", "steer_RL",
            "steer_RR", "T_FL", "T_FR", "T_RL", "T_RR", "My_FL", "My_FR", "My_RL", "My_RR", "rsteer_FL", "rsteer_FR",
            "rsteer_RL", "rsteer_RR", "in_zone"]


def get_track(name: str, width: float) -> T.Track:
    if name.startswith("drift_"):
        return T.eval_drift_track(name, w=width)
    if name == "race_A":
        return T.race_circuit(w=width)
    if name.startswith("dev_"):
        # development courses for pilots, so the protocol-107 test tracks stay unseen: a random course from a
        # fixed seed (the same generator as training, but seeds >= 9000 are never used by actors)
        tr = T.random_course(np.random.default_rng(int(name[4:])), w=width)
        tr.name = name
        return tr
    raise ValueError(name)


def run_episode(env, pol, track, mu, v0, mu_hat=None):
    # fresh=True: a new CarSim session per evaluation episode, so every arm starts from the identical state;
    # with cfg.probe the friction probe runs first (all arms) and sets env.mu_hat
    obs = env.reset(track=track, mu=mu, v0_kph=v0, mu_hat=mu_hat, fresh=True)
    zone = zone_ok = 0
    zone_beta = []
    while True:
        a = pol.act(torch.from_numpy(obs)[None], deterministic=True).numpy()[0]
        obs, r, term, trunc, info = env.step(a)
        if info["in_zone"] > 0:
            zone += 1
            zone_ok += abs(info["beta"] - info["beta_ref"]) <= np.radians(5.0)
            zone_beta.append(abs(info["beta"]))
        if term or trunc:
            break
    L = np.array(env.log)
    m = dict(track=track.name, mu=mu, v0=v0, success=bool(info["success"]), reason=info["reason"] or "trunc",
             t=float(info["t"]), progress=float(info["s_prog"]), length=float(track.length),
             mean_v=float(np.mean(L[:, 4])), mean_abs_elat=float(np.mean(np.abs(L[:, 8]))),
             max_abs_elat=float(np.max(np.abs(L[:, 8]))), max_abs_beta_deg=float(np.degrees(np.max(np.abs(L[:, 7])))),
             zone_steps=int(zone), zone_ok_frac=float(zone_ok / zone) if zone else None,
             zone_mean_abs_beta_deg=float(np.degrees(np.mean(zone_beta))) if zone_beta else None,
             lap_time=float(info["t"]) if (track.closed and info["success"]) else None,
             mu_hat=None if env.mu_hat is None else float(env.mu_hat), probe=env.probe_info)
    return m, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--mu", default="0.4,0.7,0.95")
    ap.add_argument("--v0", type=float, default=60.0)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--weights", default="actor_final.pt")
    a = ap.parse_args()
    run = Path(a.run)
    cfgj = json.loads((run / "config.json").read_text())
    ecfg = E.EnvConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfgj["env"].items()})
    ecfg.record = True
    ecfg.record_raw = False
    if ecfg.friction_info.startswith("estimate") and not ecfg.probe:
        raise SystemExit("estimate arm without the probe: mu_hat would be undefined (never substitute the true mu)")
    tracks = a.tracks.split(",") if a.tracks else (["drift_A", "drift_B"] if ecfg.task == "drift" else ["race_A"])
    pol = Actor(cfgj["n_obs"], cfgj["n_act"], cfgj["sac"]["hidden"], cfgj["sac"]["layers"])
    pol.load_state_dict(torch.load(run / a.weights, map_location="cpu", weights_only=True))
    pol.eval()
    env = E.DDEVEnv(ecfg, run / "carsim" / f"{a.tag}", seed=0)
    rows, trajs = [], {}
    for tn in tracks:
        for mu in [float(x) for x in a.mu.split(",")]:
            tr = get_track(tn, ecfg.width)
            m, L = run_episode(env, pol, tr, mu, a.v0)
            rows.append(m)
            trajs[f"{tn}_mu{mu:.2f}"] = L
            print(json.dumps(m), flush=True)
    env.close()
    np.savez_compressed(run / f"{a.tag}_trajectories.npz", columns=np.array(LOG_COLS), **trajs)
    (run / f"{a.tag}.json").write_text(json.dumps(dict(run=str(run), weights=a.weights, rows=rows), indent=1))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
