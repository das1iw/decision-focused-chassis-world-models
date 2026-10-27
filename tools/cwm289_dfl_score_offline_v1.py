"""Offline decision fidelity of any drifting CWM variant on stored twin decisions (research/289 step 2).

Input: decision files <out>_dfl.npz written by tools/cwm289_branch_cl_v5.py --dfl_sub M (twin arm, --save_values): per
decision the complete planning inputs and M candidate action sequences (the twin's M/2 best and M/2 random others) with
their twin values. Every model listed scores the SAME candidates with the planner objective in torch
(trfc_drift/torch_objective289_v2, checked against the CUDA kernel, 07 2026-10-15 01:14), so models that did not exist
when the twin ran (e.g. decision-focused residuals) are compared on equal terms.

    .venv-wm/Scripts/python.exe -B tools/cwm289_dfl_score_offline_v1.py --dfl D:/.../twb_dfl_nom_dfl.npz --courses 9040,9041,9042 \
        --models physics=physics,mlp3=resid:D:/.../resid_v3/mlp.pt,pure=pure:D:/.../pure_gru.pt --out experiments/results/wm4_289/x.json

Models: name=physics (planar physics v2), name=resid:<checkpoint> (residual port, MLP or GRU; GRU memory from the stored
10-step observed history as in training), name=pure:<checkpoint> (matched-data generic GRU, memory from the same history).
Metrics per decision (lambda = the planner's): rank (Spearman over the M candidates), regretW (twin value of the twin's
MPPI weights minus twin value of the model's weights), regret1 (twin value lost by the model's best candidate), du0
(max-norm difference of the weighted first actions, normalized units), ce (cross-entropy of the model's weights against the
twin's), top8 (share of the model's 8 best candidates among the twin's 8 best), ffw (model weight on twin-dead
candidates), ae (relative action-effect error over up to 256 random pairs of twin-alive candidates:
RMS[(Gm_i - Gm_j) - (Gtw_i - Gtw_j)] / RMS[Gtw_i - Gtw_j]). Means over decisions, per course, and paired differences
against the reference model (--ref, default physics) with an exact sign-flip test over courses (all 2^C sign patterns
when C <= 16) on per-course mean differences.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trfc_drift import mppi_cuda289_resid_v2 as RK  # noqa: E402
from trfc_drift.torch_objective289_v2 import objective, physics_step, pure_step  # noqa: E402
from trfc_drift.wm_physics2_resid289_v1 import ResidPhysics2  # noqa: E402

METRICS = ('rank', 'regretW', 'regret1', 'du0', 'ce', 'top8', 'ffw', 'ae')


def load_dfl(paths):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    n0 = parts[0]['A'].shape[0]
    keys = [k for k in parts[0].files if parts[0][k].ndim >= 1 and parts[0][k].shape[0] == n0]
    D = {k: np.concatenate([z[k] for z in parts]) for k in keys}
    D['eP'] = parts[0]['eP']; D['lam'] = float(parts[0]['lam'])
    return D


def make_model(spec, dev, dt):
    kind, _, path = spec.partition(':')
    if kind == 'physics':
        return dict(kind='physics', m=ResidPhysics2(resid=None, device=dev, dtype=dt, n_sub=5))
    if kind == 'resid':
        r, sc = RK.load_resid(path)
        r = r.to(dev).to(dt)
        return dict(kind='resid', m=ResidPhysics2(resid=r, scale=sc, device=dev, dtype=dt, n_sub=5), memory=bool(getattr(r, 'memory', False)))
    if kind == 'pure':
        spec_ = importlib.util.spec_from_file_location('pt', ROOT / 'tools' / 'cwm289_pure_train_v1.py')
        pmod = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(pmod)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        pm = pmod.PureGRU(ck['hidden'], ck['width']); pm.load_state_dict(ck['state']); pm.eval()
        return dict(kind='pure', m=pm.to(dev))
    raise ValueError(spec)


@torch.no_grad()
def values(mod, D, idx, dev, dt):
    f = lambda k: torch.as_tensor(D[k][idx], dtype=dt, device=dev)  # noqa: E731
    A = D['A'][idx]; B, M, T, na = A.shape
    rows = torch.arange(B, device=dev).repeat_interleave(M)
    W = dict(txy=f('txy'), tpsi=f('tpsi'), tsw=f('tsw'), bref=f('bref'), vl=f('vl'), s0p=f('s0p'))
    At = torch.as_tensor(A, dtype=dt, device=dev).reshape(B * M, T, na)
    common = (At, f('mu')[rows], D['eP'], W, rows, f('prev_a'), f('st0'), f('ts0'), f('pose0'))
    if mod['kind'] in ('physics', 'resid'):
        z = None
        if mod['kind'] == 'resid' and mod['memory']:
            hs, hh, hu, hm = f('hist_s'), f('hist_h'), f('hist_u'), f('hist_mu')
            for j in range(hs.shape[1]):
                _, _, z = mod['m'].step(hs[:, j], hh[:, j], hu[:, j], hm[:, j], z=z)
            z = z[rows]
        G, alive = objective(physics_step(mod['m']), (f('h0')[rows], z), f('s0')[rows], *common, dtype=dt)
    else:
        zp = None
        hs, hu, hm = (torch.as_tensor(D[k][idx], dtype=torch.float32, device=dev) for k in ('hist_s', 'hist_u', 'hist_mu'))
        for j in range(hs.shape[1]):
            _, zp = mod['m'].step(hs[:, j], hu[:, j], hm[:, j], zp)
        G, alive = objective(pure_step(mod['m']), None if zp is None else zp[rows], f('s0')[rows], *common, dtype=dt)
    return G.reshape(B, M).double().cpu().numpy(), alive.reshape(B, M).cpu().numpy()


def spearman(a, b):
    ra = np.argsort(np.argsort(a, 1), 1).astype(float); rb = np.argsort(np.argsort(b, 1), 1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-30)


def softw(G, lam):
    w = np.exp((G - G.max(1, keepdims=True)) / lam)
    return w / w.sum(1, keepdims=True)


def metrics(Gm, Gt, alive_t, A0, lam, rng):
    wt, wm = softw(Gt, lam), softw(Gm, lam)
    out = dict(rank=spearman(Gm, Gt), regretW=(wt * Gt).sum(1) - (wm * Gt).sum(1),
               regret1=Gt.max(1) - Gt[np.arange(len(Gt)), Gm.argmax(1)],
               du0=np.abs((wm[..., None] * A0).sum(1) - (wt[..., None] * A0).sum(1)).max(-1),
               ce=-(wt * np.log(np.maximum(wm, 1e-300))).sum(1), ffw=(wm * (alive_t < 0.5)).sum(1))
    t8m = np.argsort(-Gm, 1)[:, :8]; t8t = np.argsort(-Gt, 1)[:, :8]
    out['top8'] = np.array([len(set(a_) & set(b_)) / 8.0 for a_, b_ in zip(t8m, t8t)])
    ae = np.full(len(Gt), np.nan)
    for i in range(len(Gt)):
        al = np.flatnonzero(alive_t[i] > 0.5)
        if len(al) < 3:
            continue
        pairs = np.array(list(itertools.combinations(al, 2)))
        if len(pairs) > 256:
            pairs = pairs[rng.choice(len(pairs), 256, replace=False)]
        dt_ = Gt[i, pairs[:, 0]] - Gt[i, pairs[:, 1]]; dm = Gm[i, pairs[:, 0]] - Gm[i, pairs[:, 1]]
        den = np.sqrt((dt_ ** 2).mean())
        if den > 1e-9:
            ae[i] = np.sqrt(((dm - dt_) ** 2).mean()) / den
    out['ae'] = ae
    return out


def signflip(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n == 0:
        return float('nan')
    obs = abs(x.mean())
    if n <= 16:
        tot = 0; cnt = 0
        for s in itertools.product((1.0, -1.0), repeat=n):
            tot += 1; cnt += abs((np.array(s) * x).mean()) >= obs - 1e-15
        return cnt / tot
    rng = np.random.default_rng(0); s = rng.choice((1.0, -1.0), (20000, n))
    return float((np.abs((s * x).mean(1)) >= obs - 1e-15).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dfl', required=True)
    ap.add_argument('--courses', default='heldout', help="'heldout' (last 3 sorted courses), 'all' or a comma list")
    ap.add_argument('--models', required=True)
    ap.add_argument('--ref', default='physics')
    ap.add_argument('--dtype', default='float64', choices=('float32', 'float64'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--max_n', type=int, default=0, help='0: all selected decisions; else a fixed random subset')
    ap.add_argument('--save_per', default='', help='npz path: per-decision metrics of every model (added 2026-10-15 10:3x)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dt = torch.float32 if a.dtype == 'float32' else torch.float64
    D = load_dfl(a.dfl.split(',')); lam = D['lam']
    cs = sorted(set(D['course'].tolist()))
    sel_c = cs[-3:] if a.courses == 'heldout' else (cs if a.courses == 'all' else [int(c) for c in a.courses.split(',')])
    idx = np.flatnonzero(np.isin(D['course'], sel_c))
    if a.max_n and a.max_n < len(idx):
        idx = np.sort(np.random.default_rng(7).choice(idx, a.max_n, replace=False))
    Gt = D['Gtw'][idx].astype(float); alt = D['alive_tw'][idx].astype(float); A0 = D['A'][idx][:, :, 0, :].astype(float)
    course = D['course'][idx]
    rep = dict(args=vars(a), n=int(len(idx)), courses=sel_c, models={}, per_course={}, paired={})
    per = {}
    for item in a.models.split(','):
        name, spec = item.split('=', 1)
        t0 = time.time(); mod = make_model(spec, a.device, dt)
        Gm = np.concatenate([values(mod, D, idx[i:i + a.bs], a.device, dt)[0] for i in range(0, len(idx), a.bs)])
        m = metrics(Gm, Gt, alt, A0, lam, np.random.default_rng(11)); per[name] = m
        rep['models'][name] = {k: float(np.nanmean(v)) for k, v in m.items()}
        rep['models'][name]['wall_s'] = time.time() - t0
        rep['per_course'][name] = {int(c): {k: float(np.nanmean(v[course == c])) for k, v in m.items()} for c in sel_c}
        print(name, json.dumps({k: round(v, 4) for k, v in rep['models'][name].items()}), flush=True)
    if a.ref in per:
        for name in per:
            if name == a.ref:
                continue
            d = {}
            for k in METRICS:
                diff = per[name][k] - per[a.ref][k]
                cm = [float(np.nanmean(diff[course == c])) for c in sel_c]
                d[k] = dict(mean=float(np.nanmean(diff)), course_means=cm, signflip_p=signflip(cm),
                            courses_better=int(sum((x > 0) if k in ('rank', 'top8') else (x < 0) for x in cm)))
            rep['paired'][f'{name}-{a.ref}'] = d
    for name, m in per.items():
        rep['models'][name]['quantiles'] = {k: {str(q): float(np.nanquantile(v, q)) for q in (0.5, 0.9, 0.99)} for k, v in m.items()}
    if a.save_per:
        np.savez_compressed(a.save_per, idx=idx, course=course, **{f'{n}__{k}': v for n, m in per.items() for k, v in m.items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
