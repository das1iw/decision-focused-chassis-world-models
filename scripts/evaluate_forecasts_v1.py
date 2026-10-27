"""Evaluate forecasts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cwm import residual_kernel_v1 as RK  # noqa: E402  (load_resid, gru_memory; no GPU use here)
from cwm.residual_v1 import ResidualPhysics  # noqa: E402

DT = 0.02
HS = (('0.1s', 5), ('0.2s', 10), ('0.5s', 25), ('1.0s', 50), ('1.5s', 75))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load(path):
    z = np.load(path)
    tags = sorted({k.split('_')[0] for k in z.files})
    n = min(z[f'{t}_sig'].shape[1] for t in tags)
    D = {k: [] for k in ('sig', 'u', 'h', 'mu', 'live', 'pose', 'course', 't_drop', 'mu0', 'margin')}
    for t in tags:
        for k in ('sig', 'u', 'h', 'mu', 'live', 'pose'):
            D[k].append(z[f'{t}_{k}'][:, :n])
        E = z[f'{t}_sig'].shape[0]
        D['course'].append(z[f'{t}_course']); D['t_drop'].append(z[f'{t}_t_drop']); D['mu0'].append(z[f'{t}_mu0'])
        D['margin'].append(np.full(E, int(t[1:]) / 100.0))
    return {k: np.concatenate(v) for k, v in D.items()}


def windows(D, eps, W, H, stride):
    out = []
    for e in eps:
        live = D['live'][e]
        for k in range(W, live.shape[0] - H - 1, stride):
            if live[k - W:k + H + 1].min() > 0.5:
                out.append((e, k))
    return out


def integrate_pose(pose0, V):
    """pose0 [B, 3]; V [B, H+1, 3] velocities (vx, vy, r) at steps 0..H -> poses [B, H, 3] after each step
    (physics_kernel_v1 midpoint rule)."""
    x, y, yaw = pose0[:, 0].copy(), pose0[:, 1].copy(), pose0[:, 2].copy()
    out = []
    for j in range(V.shape[1] - 1):
        vxm = 0.5 * (V[:, j, 0] + V[:, j + 1, 0]); vym = 0.5 * (V[:, j, 1] + V[:, j + 1, 1]); rm = 0.5 * (V[:, j, 2] + V[:, j + 1, 2])
        ym = yaw + 0.5 * DT * rm
        x = x + DT * (vxm * np.cos(ym) - vym * np.sin(ym)); y = y + DT * (vxm * np.sin(ym) + vym * np.cos(ym)); yaw = yaw + DT * rm
        out.append(np.stack([x, y, yaw], 1))
    return np.stack(out, 1)


@torch.no_grad()
def predict(name, mdl, D, idx, W, H, bs=512):
    """Returns predicted states [N, H, 11] (float64 numpy)."""
    out = []
    for i in range(0, len(idx), bs):
        sub = idx[i:i + bs]
        e = np.array([q[0] for q in sub]); k = np.array([q[1] for q in sub])
        kk = k[:, None] + np.arange(-W, H + 1)[None]
        f = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
        sig = f(D['sig'][e[:, None], kk]); u = f(D['u'][e[:, None], kk]); h = f(D['h'][e[:, None], kk]); mu = f(D['mu'][e[:, None], kk])
        if name == 'pure':
            m = mdl
            sigf, uf, muf = sig.float(), u.float(), mu.float()
            z = None
            for j in range(W):
                _, z = m.step(sigf[:, j, :11], uf[:, j], muf[:, j], z)
            s = sigf[:, W, :11]; ps = []
            for j in range(H):
                s, z = m.step(s, uf[:, W + j], muf[:, W + j], z)
                ps.append(s.double())
            out.append(torch.stack(ps, 1).numpy())
            continue
        z = None
        if getattr(mdl.resid, 'memory', False):
            for j in range(W):
                _, _, z = mdl.step(sig[:, j, :11], h[:, j], u[:, j], mu[:, j], z=z)
        s, hh = sig[:, W, :11], h[:, W]; ps = []
        for j in range(H):
            s, hh, z = mdl.step(s, hh, u[:, W + j], mu[:, W + j], z=z)
            ps.append(s)
        out.append(torch.stack(ps, 1).numpy())
    return np.concatenate(out)


def boot_ci(per_ep, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(per_ep, float)
    b = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--mlp', default='')
    ap.add_argument('--gru', default='')
    ap.add_argument('--pure', default='', help='comma-separated pure GRU checkpoints (name=path or path)')
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    D = load(a.data)
    courses = sorted(set(D['course'].tolist())); val_c = courses[-4:]
    va_e = [e for e in range(len(D['course'])) if D['course'][e] in val_c]
    W, Hmax = 10, 75
    idx = windows(D, va_e, W, Hmax, a.stride)
    e_ix = np.array([q[0] for q in idx]); k_ix = np.array([q[1] for q in idx])
    models = {'physics': ResidualPhysics(resid=None, device='cpu', dtype=torch.float64, n_sub=5)}
    if a.mlp:
        r, sc = RK.load_resid(a.mlp); models['mlp'] = ResidualPhysics(resid=r, scale=sc, device='cpu', dtype=torch.float64, n_sub=5)
    if a.gru:
        r, sc = RK.load_resid(a.gru); models['gru'] = ResidualPhysics(resid=r, scale=sc, device='cpu', dtype=torch.float64, n_sub=5)
    if a.pure:
        pm = _mod('train_neural_baseline_v1')
        for spec in a.pure.split(','):
            nm, p = spec.split('=') if '=' in spec else ('pure', spec)
            ck = torch.load(p, map_location='cpu', weights_only=False)
            m = pm.PureGRU(ck['hidden'], ck['width']); m.load_state_dict(ck['state']); m.eval()
            models[nm] = ('pure', m)
    # reference quantities
    kk = k_ix[:, None] + np.arange(0, Hmax + 1)[None]
    S_true = D['sig'][e_ix[:, None], kk][:, :, :11].astype(np.float64)          # steps 0..H
    pose0 = D['pose'][e_ix, k_ix].astype(np.float64)
    P_ref = integrate_pose(pose0, S_true[:, :, :3])
    beta = lambda S: np.degrees(np.arctan2(S[..., 1], np.maximum(np.abs(S[..., 0]), 0.5)))  # noqa: E731
    t_start = k_ix * DT; t_drop = D['t_drop'][e_ix]
    strata = {
        'all': np.ones(len(idx), bool),
        'margin_0.9': D['margin'][e_ix] == 0.9, 'margin_0.75': D['margin'][e_ix] == 0.75,
        'nominal': ~np.isfinite(t_drop) | (t_start + Hmax * DT <= t_drop),
        'drop_transient': np.isfinite(t_drop) & (t_start + Hmax * DT > t_drop) & (t_start <= t_drop + 3.0),
        'beta_gt15': np.abs(beta(S_true[:, 0])) > 15.0,
    }
    for m0 in (0.35, 0.55, 0.75, 0.95):
        strata[f'mu_{m0}'] = np.abs(D['mu0'][e_ix] - m0) < 1e-6
    res = dict(n_windows=len(idx), val_courses=val_c, n_val_eps=len(va_e), strata_n={k: int(v.sum()) for k, v in strata.items()},
               models={}, paired={})
    per_win = {}
    for name, mdl in models.items():
        tm0 = time.time()
        if isinstance(mdl, tuple):
            S = predict('pure', mdl[1], D, idx, W, Hmax)
        else:
            S = predict(name, mdl, D, idx, W, Hmax)
        V = np.concatenate([S_true[:, :1, :3], S[:, :, :3]], 1)
        P = integrate_pose(pose0, V)
        rep = {}
        pw = {}
        for Hs, H in HS:
            err = S[:, H - 1, :7] - S_true[:, H, :7]
            dpos = np.hypot(P[:, H - 1, 0] - P_ref[:, H - 1, 0], P[:, H - 1, 1] - P_ref[:, H - 1, 1])
            dyaw = np.degrees(np.abs(np.angle(np.exp(1j * (P[:, H - 1, 2] - P_ref[:, H - 1, 2])))))
            dbeta = np.abs(beta(S[:, H - 1]) - beta(S_true[:, H]))
            nonfinite = ~np.isfinite(S[:, H - 1, :7]).all(1)
            q = dict(vx2=err[:, 0] ** 2, vy2=err[:, 1] ** 2, r2=err[:, 2] ** 2, om2=(err[:, 3:7] ** 2).mean(1), pos=dpos, yaw=dyaw,
                     beta=dbeta)
            pw[Hs] = q
            rep[Hs] = {}
            for sn, msk in strata.items():
                ok = msk & ~nonfinite
                if ok.sum() == 0:
                    continue
                rep[Hs][sn] = dict(n=int(msk.sum()), nonfinite=int((msk & nonfinite).sum()),
                                   rmse_vx=float(np.sqrt(q['vx2'][ok].mean())), rmse_vy=float(np.sqrt(q['vy2'][ok].mean())),
                                   rmse_r=float(np.sqrt(q['r2'][ok].mean())), rmse_omega=float(np.sqrt(q['om2'][ok].mean())),
                                   pos_median=float(np.median(q['pos'][ok])), pos_p90=float(np.quantile(q['pos'][ok], 0.9)),
                                   yaw_median_deg=float(np.median(q['yaw'][ok])), beta_median_deg=float(np.median(q['beta'][ok])),
                                   beta_p90_deg=float(np.quantile(q['beta'][ok], 0.9)))
        res['models'][name] = rep; per_win[name] = pw
        print(f'{name}: {time.time() - tm0:.0f} s; 1.0 s all ' + json.dumps({k: round(v, 4) for k, v in rep['1.0s']['all'].items()}),
              flush=True)
    # paired differences vs physics and vs the generic learned model, per episode cluster
    names = list(models)
    for ref in [n for n in ('physics', 'pure') if n in names]:
        for name in names:
            if name == ref:
                continue
            for Hs, _ in HS:
                for metric in ('pos', 'beta', 'om2', 'vy2'):
                    d = per_win[name][Hs][metric] - per_win[ref][Hs][metric]
                    ok = np.isfinite(d)
                    eps_u = np.unique(e_ix[ok])
                    per_ep = np.array([d[ok & (e_ix == e)].mean() for e in eps_u])
                    res['paired'][f'{name}-{ref}|{Hs}|{metric}'] = dict(mean=float(per_ep.mean()), ci95=boot_ci(per_ep),
                                                                        frac_eps_better=float((per_ep < 0).mean()),
                                                                        n_eps=int(len(eps_u)))
    res['wall_s'] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    lines = ['| model | horizon | pos median (m) | pos p90 | beta median (deg) | vy RMSE | r RMSE | omega RMSE |', '|---|---|---|---|---|---|---|---|']
    for name in names:
        for Hs, _ in HS:
            q = res['models'][name][Hs]['all']
            lines.append(f"| {name} | {Hs} | {q['pos_median']:.3f} | {q['pos_p90']:.3f} | {q['beta_median_deg']:.2f} | "
                         f"{q['rmse_vy']:.3f} | {q['rmse_r']:.4f} | {q['rmse_omega']:.2f} |")
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
