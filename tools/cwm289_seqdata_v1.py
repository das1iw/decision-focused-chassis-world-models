"""Causal sequence dataset for the drifting CWM (research/289 step 2, Codex review 292 order 2): the 44-state twin as the
plant, driven by the 284 drift planner (planar physics-v2 MPPI + System 2), with everything the planar planning model sees
and what the plant did, per 20 ms control step. Development data on courses NOT used by the 289 branching tests.

    .venv-wm/Scripts/python.exe -B trfc_xcar/run_pinned284.py --cores 28-31 tools/cwm289_seqdata_v1.py \
        --courses 9027-9042 --out D:/TRFC_data/wm4_289/seqdata_v1/data.npz

Episodes: courses x friction .35/.55/.75/.95 x condition {nominal, drop (x0.7 at a per-episode random time 6-14 s)} x
margin {0.9, 0.75}; from s = 0 at 60 km/h; t_limit (default 40 s); K 512. Per step and episode (float32):
  sig  [vx, vy, r, omega x4, torque x4, ax, ay]  plant signals BEFORE the step (the planar observer's state source)
  pose [x, y, yaw]
  u    [steer x4 (road-wheel, rad), motor x4 (N m)]  kernel commands applied over the step
  h    planar observer hidden state (18) after observing the pre-step signals (the planner's h0)
  mu   true road friction (per wheel mean) during the step; mu_in the planner's friction input
  live 1 while the episode is running (crash/finish ends it)
Separate namespace: no privileged quantity is an input of the planar model; the plant's 44-state values are not saved.
Stop gracefully with <out>.STOP.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp  # noqa: F401  (planner)
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MUFILTER = dict(window=5, forget=0.9, feed='q_lo')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--courses', default='9027-9042')
    ap.add_argument('--mus', default='0.35,0.55,0.75,0.95')
    ap.add_argument('--margins', default='0.9,0.75')
    ap.add_argument('--K', type=int, default=512)
    ap.add_argument('--t_limit', type=float, default=40.0)
    ap.add_argument('--seed', type=int, default=2893)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from trfc_drift.chassis_env246_v1 import EnvConfig246
    from trfc_drift.drift_mppi_twin284_v1 import TwinDriftMPPI
    from trfc_drift.ppo246_v1 import course, make_env, plant_and_base
    lo, hi = (int(x) for x in a.courses.split('-'))
    courses = list(range(lo, hi + 1)); tracks = [course(s) for s in courses]; n_c = len(tracks)
    mus = [float(x) for x in a.mus.split(',')]; margins = [float(x) for x in a.margins.split(',')]
    rng = np.random.default_rng(a.seed)
    cfg = EnvConfig246(random_start=False, t_max=a.t_limit)
    pb = plant_and_base()
    out = {}
    t_all = time.perf_counter()
    for vm in margins:
        E = n_c * len(mus) * 2                     # x conditions nominal/drop
        env = make_env(E, cfg, tracks, 7, pb); dev = env.dev
        tid = torch.arange(E, device=dev) % n_c
        mu_ep = np.array([mus[(e // n_c) % len(mus)] for e in range(E)])          # rows: (condition, friction, course)
        drop_ep = np.array([e >= E // 2 for e in range(E)])
        t_drop = np.where(drop_ep, rng.uniform(6.0, 14.0, E), np.inf)
        mu = torch.as_tensor(mu_ep, dtype=env.dtype, device=dev)
        env.reset(tid=tid, s0=torch.zeros(E, dtype=env.dtype, device=dev), v0=torch.full((E,), 60 / 3.6, dtype=env.dtype, device=dev),
                  mu=mu, mu_hat=mu.clone())
        ctl = TwinDriftMPPI(E, K=a.K, T=75, seed=a.seed + int(vm * 100), v_margin=vm, mufilter=MUFILTER)
        mu_hat = mu.double().cpu().numpy()
        n_steps = int(round(a.t_limit / cfg.dt))
        rec = {k: np.zeros((E, n_steps) + sh, np.float32) for k, sh in
               (('sig', (13,)), ('pose', (3,)), ('u', (8,)), ('h', (18,)), ('mu', ()), ('mu_in', ()), ('live', ()))}
        done = torch.zeros(E, dtype=torch.bool, device=dev); dropped = np.zeros(E, bool); t0 = time.perf_counter()
        for k in range(n_steps):
            t = k * cfg.dt
            newd = (~dropped) & (t >= t_drop)
            if newd.any():
                idx = torch.as_tensor(np.flatnonzero(newd), device=dev)
                env.kernel.mu[idx] = env.kernel.mu[idx] * 0.7; env.mu[idx] = env.mu[idx] * 0.7; dropped |= newd
            act = ctl.act(env, mu_hat)
            sgn = ctl._signals(env)
            rec['sig'][:, k] = np.concatenate([np.stack([sgn['vx'], sgn['vy'], sgn['r']], 1), sgn['omega'], sgn['torque'],
                                               np.stack([sgn['ax'], sgn['ay']], 1)], 1)
            rec['pose'][:, k] = np.stack([sgn['x'], sgn['y'], sgn['yaw']], 1)
            rec['h'][:, k] = np.stack(ctl.hn)
            rec['mu'][:, k] = env.mu[:, 0].double().cpu().numpy()
            rec['mu_in'][:, k] = ctl.mu_est if ctl.mu_est is not None else mu_hat
            rec['live'][:, k] = (~done).cpu().numpy()
            o, p, r, term, trunc, info = env.step(act, auto_reset=False)
            rec['u'][:, k] = np.concatenate([env.kernel.steer.double().cpu().numpy(), env.kernel.motor.double().cpu().numpy()], 1)
            done |= term | trunc
            if k % 250 == 0:
                print(f'  margin {vm} t {t:5.1f} s: done {int(done.sum())}/{E}, wall {time.perf_counter() - t0:.0f} s', flush=True)
            if bool(done.all()) or Path(a.out + '.STOP').exists():
                rec = {kk: v[:, :k + 1] for kk, v in rec.items()}
                break
        tag = f'm{int(round(vm * 100))}'
        for kk, v in rec.items():
            out[f'{tag}_{kk}'] = v
        out[f'{tag}_course'] = np.array([courses[e % n_c] for e in range(E)]); out[f'{tag}_mu0'] = mu_ep
        out[f'{tag}_t_drop'] = t_drop
        if Path(a.out + '.STOP').exists():
            break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    meta = dict(args=vars(a), wall_s=time.perf_counter() - t_all, keys=sorted(out), note='u is the command applied over step k; '
                'sig/h are before step k; next-step plant signals are sig[:, k+1]')
    Path(a.out.replace('.npz', '.json')).write_text(json.dumps(meta, indent=1))
    print('wrote', a.out, json.dumps({k: list(v.shape) for k, v in out.items() if k.endswith('_sig')}))


if __name__ == '__main__':
    main()
