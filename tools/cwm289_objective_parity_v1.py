"""Parity of the torch planner objective (trfc_drift/torch_objective289_v1) with the CUDA kernels on real branch states
(research/289 step 2): the same candidate sets scored by
  - the original planar kernel (mppi_cuda284_f32_v1.rollout_gpu32, float32) vs torch with planar physics v2 (float64);
  - the residual-port kernel (mppi_cuda289_resid_v2, float32) vs torch with the same residual (float64), if --mlp given.
Pass: median |dG| / (|G| spread) <= 1e-3 and Spearman rank correlation >= 0.999 per decision, survival flags agree on
>= 99.5 % of candidates. A dry first decision of a few branch points; development check of the implementation.

    .venv-wm/Scripts/python.exe -B trfc_xcar/run_pinned284.py --cores 28-31 tools/cwm289_objective_parity_v1.py \
        --branches D:/TRFC_data/wm4_289/branch_v3/m75/branches.pkl --mlp D:/TRFC_data/wm4_289/resid_v1/mlp.pt \
        --out experiments/results/wm4_289/objective_parity_v1.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def spearman(a, b):
    ra = np.argsort(np.argsort(a, 1), 1).astype(float); rb = np.argsort(np.argsort(b, 1), 1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branches', required=True)
    ap.add_argument('--mlp', default='')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--K', type=int, default=256)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from trfc_drift.chassis_env246_v1 import EnvConfig246
    from trfc_drift.drift_mppi_twin284_v1 import TwinDriftMPPI
    from trfc_drift.ppo246_v1 import course, make_env, plant_and_base
    from trfc_drift.system2_246_v1 import load
    from trfc_drift.mppi_cuda284_f32_v1 import rollout_gpu32
    from trfc_drift import mppi_cuda289_resid_v2 as RK
    from trfc_drift.torch_objective289_v1 import objective, physics_step
    from trfc_drift.twin_cost289_v1 import window_tensors
    from trfc_drift.wm_physics2_resid289_v1 import ResidPhysics2
    with open(a.branches, 'rb') as fh:
        B = pickle.load(fh)
    meta, bps = B['meta'], B['branches']
    bps = [bps[i] for i in np.linspace(0, len(bps) - 1, a.n).astype(int)]
    courses = [int(c) for c in range(int(meta['args']['courses'].split('-')[0]), int(meta['args']['courses'].split('-')[1]) + 1)]
    Eb, K, T = len(bps), a.K, meta['args']['T']
    env = make_env(Eb, EnvConfig246(random_start=False, t_max=30.0), [course(c) for c in courses], 7, plant_and_base())
    dev = env.dev
    env.reset(tid=torch.as_tensor([courses.index(bp['course']) for bp in bps], device=dev), s0=torch.zeros(Eb, dtype=env.dtype, device=dev),
              v0=torch.full((Eb,), 60 / 3.6, dtype=env.dtype, device=dev), mu=torch.full((Eb,), 0.75, dtype=env.dtype, device=dev),
              mu_hat=torch.full((Eb,), 0.75, dtype=env.dtype, device=dev))
    comb = {k: torch.cat([bp['snap'][k] for bp in bps]).to(dev) for k in bps[0]['snap']}
    load(env, comb, torch.arange(Eb, device=dev), torch.arange(Eb, device=dev))
    ctl = TwinDriftMPPI(Eb, K=K, T=T, seed=3, v_margin=meta['args']['v_margin'])
    ctl.Un = cp.asarray(np.stack([bp['Un'] for bp in bps]))
    ctl.hn = [bp['hn'].copy() for bp in bps]; ctl.sn_prev = [bp['sn_prev'].copy() for bp in bps]
    ctl.u_prev = [bp['u_prev'].copy() for bp in bps]
    sg = ctl._signals(env)
    mus = np.array([bp['mu'] for bp in bps], float)
    for e in range(Eb):
        ctl.observe(sg, e, float(mus[e]))
    s_k = np.stack(ctl.sn_prev); h_k = np.stack(ctl.hn)
    win = ctl.windows(env, sg, mus)
    eps = ctl.rng.standard_normal((Eb, K, T, ctl.na)) * ctl.sig; eps[:, 0] = 0.0
    A = cp.clip(ctl.Un[:, None] + eps, -1.0, 1.0)
    pose0 = np.stack([sg['x'], sg['y'], sg['yaw']], 1)
    kargs = (A, s_k, h_k, sg['steer_state'], sg['torque_state'], pose0, sg['prev_a'], mus, ctl.ph.n_sub, ctl.nb.EP, ctl.packed,
             *win)
    rows = torch.arange(Eb, device=dev).repeat_interleave(K)
    W = window_tensors(win, dev, torch.float64)
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)  # noqa: E731
    At = torch.as_tensor(A.reshape(Eb * K, T, ctl.na), device=dev)
    rep = {}
    cases = [('physics', None)]
    if a.mlp:
        cases.append(('mlp', a.mlp))
    for name, path in cases:
        if path is None:
            Gk, alk = rollout_gpu32(*kargs); tp = ResidPhysics2(resid=None, device=dev, dtype=torch.float64, n_sub=5)
        else:
            r, sc = RK.load_resid(path); r = r.to(dev)
            Gk, alk = RK.rollout_gpu32_resid(*kargs, resid=RK.pack_resid(r), rscale=sc)
            tp = ResidPhysics2(resid=r, scale=sc, device=dev, dtype=torch.float64, n_sub=5)
        Gt, alt = objective(physics_step(tp), (f(h_k)[rows], None), f(s_k)[rows], At, f(mus)[rows], ctl.nb.EP, W, rows,
                            f(sg['prev_a']), f(sg['steer_state']), f(sg['torque_state']), f(pose0))
        Gk = cp.asnumpy(Gk).astype(float); alk = cp.asnumpy(alk) > 0.5
        Gt = Gt.reshape(Eb, K).cpu().numpy(); alt = alt.reshape(Eb, K).cpu().numpy() > 0.5
        spread = np.maximum(np.quantile(Gk, 0.9, 1) - np.quantile(Gk, 0.1, 1), 1e-6)
        rel = np.median(np.abs(Gk - Gt), 1) / spread
        rho = spearman(Gk, Gt)
        agree = float((alk == alt).mean())
        rep[name] = dict(rel_median_per_decision=rel.tolist(), rank=rho.tolist(), alive_agree=agree,
                         pass_=bool(rel.max() <= 1e-3 and rho.min() >= 0.999 and agree >= 0.995))
        print(name, 'rel max', float(rel.max()), 'rank min', float(rho.min()), 'alive agree', agree, flush=True)
    rep['pass'] = all(v['pass_'] for v in rep.values() if isinstance(v, dict))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print('pass', rep['pass'])
    sys.exit(0 if rep['pass'] else 1)


if __name__ == '__main__':
    main()
