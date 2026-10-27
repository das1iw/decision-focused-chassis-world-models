"""Graceful stop (added 2026-10-15 06:3x before any v3 run): console Ctrl+C does not reach Python processes started
from the agent's background shells (07 2026-10-15 06:3x), so the loop checks for <out>.STOP every iteration; on it,
the state (model, optimizer, RNGs, history, interim checks, iteration) is saved to <out>.partial.pt, the report is
written with stopped_early=true, and the process exits with code 3. --resume continues from <out>.partial.pt.
evaluate() casts the model values to float64 before the softmax (in a float32 CPU smoke the held-out cross-entropy
underflowed to inf; rank, regretW and du0 were unaffected).

Version 3 (method unchanged): float32 model/objective (trfc_drift/torch_objective289_v2, --dtype) and an optional
candidate subset per decision for training (--m_train: the stored twin-best half and random half are subsampled
equally). v2 ran at 15-40 s per 16x64 iteration on a shared GPU (launch-bound float64 work, 07 2026-10-15 05:0x).
Held-out metrics always use all stored candidates.

Version 2 of tools/cwm289_resid_dfl_v1.py (method unchanged): defaults sized for throughput (16 decisions per
iteration, 400 iterations, lr 1e-3) and interim held-out checks on a fixed random subset (--eval_max, default 160
decisions; the full held-out set before training and at the end). v1 measured about 10 s per 8-decision evaluation
batch on a shared GPU (07 2026-10-15 04:0x).

Decision-focused training of the tire-force residual port (research/289 step 2): the planar physics-v2 residual
model is trained so that the planner objective computed with it ranks MPPI candidates like the 44-state twin, i.e. its
MPPI weights match the twin's. Development.

    .venv-wm/Scripts/python.exe -B trfc_xcar/run_pinned284.py --cores 28-31 tools/cwm289_resid_dfl_v1.py \
        --dfl D:/TRFC_data/wm4_289/dfl_v1/m75_twb_dfl.npz,... --resid mlp --init D:/TRFC_data/wm4_289/resid_v3/mlp.pt \
        --out D:/TRFC_data/wm4_289/resid_dfl_v1/mlp.pt --report experiments/results/wm4_289/resid_dfl_v1_mlp.json

Data: <out>_dfl.npz files written by tools/cwm289_branch_cl_v5.py --dfl_sub M on TRAINING courses: per decision the
planning inputs (observer state and hidden state, actuator states, pose, friction input, track window, previous
action, 10-step observed history) and M candidate action sequences (the twin's M/2 best and M/2 random) with their twin
values. Split by whole course: the last --val_courses courses (sorted) are held out.
Loss per decision (lambda = the planner's 1.0):
  CE  = - sum_j p_tw,j log p_m,j,  p = softmax(G / lambda) over the subset       (the MPPI weights)
  DIF = mean_j ((G_m,j - mean G_m) - (G_tw,j - mean G_tw))^2 / var(G_tw)          (value differences)
  L   = CE + --w_dif DIF   (+ --w_fc x the v3 forecast loss on the sequence dataset, if --seqdata is given)
The objective is trfc_drift/torch_objective289_v1 (checked against the CUDA kernel: 07 01:14), differentiated through
the residual model. Held-out metrics per decision: Spearman rank, regretW within the subset (twin value of the
model's weights vs the twin's own), first-action difference of the subset-MPPI actions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trfc_drift import mppi_cuda289_resid_v2 as RK  # noqa: E402
from trfc_drift.torch_objective289_v2 import objective as _obj, physics_step  # noqa: E402
from trfc_drift.wm_physics2_resid289_v1 import GRUResid, MLPResid, ResidPhysics2  # noqa: E402

objective = getattr(_obj, '__wrapped__', _obj)          # the differentiable (no torch.no_grad) function


def load_dfl(paths):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    keys = [k for k in parts[0].files if parts[0][k].ndim >= 1 and k not in ('eP',)]
    D = {k: np.concatenate([z[k] for z in parts]) for k in keys if parts[0][k].shape[0] == parts[0]['A'].shape[0]}
    D['eP'] = parts[0]['eP']; D['lam'] = float(parts[0]['lam'])
    return D


def spearman(a, b):
    ra = np.argsort(np.argsort(a, -1), -1).astype(float); rb = np.argsort(np.argsort(b, -1), -1).astype(float)
    ra -= ra.mean(-1, keepdims=True); rb -= rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1) + 1e-30)


DT_ = torch.float32


def batch_values(model, D, idx, dev, grad=True, cand=None):
    """Model values [B, M] of the stored candidate subsets of decisions idx (cand: candidate indices to use)."""
    f = lambda k: torch.as_tensor(D[k][idx], dtype=DT_, device=dev)  # noqa: E731
    A_np = D['A'][idx] if cand is None else D['A'][idx][:, cand]
    B, M, T, na = A_np.shape
    rows = torch.arange(B, device=dev).repeat_interleave(M)
    W = dict(txy=f('txy'), tpsi=f('tpsi'), tsw=f('tsw'), bref=f('bref'), vl=f('vl'), s0p=f('s0p'))
    z = None
    if getattr(model.resid, 'memory', False) and 'hist_s' in D:
        hs, hh, hu, hm = f('hist_s'), f('hist_h'), f('hist_u'), f('hist_mu')
        for j in range(hs.shape[1]):
            _, _, z = model.step(hs[:, j], hh[:, j], hu[:, j], hm[:, j], z=z)
        z = z[rows]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        G, alive = objective(physics_step(model), (f('h0')[rows], z), f('s0')[rows],
                             torch.as_tensor(A_np, dtype=DT_, device=dev).reshape(B * M, T, na),
                             f('mu')[rows], D['eP'], W, rows, f('prev_a'), f('st0'), f('ts0'), f('pose0'), dtype=DT_)
    return G.reshape(B, M), alive.reshape(B, M)


def losses(Gm, Gt, lam, w_dif):
    pt = torch.softmax(Gt / lam, -1); lpm = torch.log_softmax(Gm / lam, -1)
    ce = -(pt * lpm).sum(-1)
    dm = Gm - Gm.mean(-1, keepdim=True); dt = Gt - Gt.mean(-1, keepdim=True)
    dif = ((dm - dt) ** 2).mean(-1) / (dt.var(-1) + 1e-6)
    return ce.mean() + w_dif * dif.mean(), ce.mean(), dif.mean()


@torch.no_grad()
def evaluate(model, D, idx, dev, lam, bs=8):
    rk, rg, du, ce_l = [], [], [], []
    for i in range(0, len(idx), bs):
        sub = idx[i:i + bs]
        Gm, _ = batch_values(model, D, sub, dev, grad=False)
        Gm = Gm.cpu().numpy().astype(float); Gt = D["Gtw"][sub].astype(float); A0 = D['A'][sub][:, :, 0, :].astype(float)
        wt = np.exp((Gt - Gt.max(-1, keepdims=True)) / lam); wt /= wt.sum(-1, keepdims=True)
        wm = np.exp((Gm - Gm.max(-1, keepdims=True)) / lam); wm /= wm.sum(-1, keepdims=True)
        rk.append(spearman(Gm, Gt)); rg.append((wt * Gt).sum(-1) - (wm * Gt).sum(-1))
        du.append(np.abs((wm[..., None] * A0).sum(1) - (wt[..., None] * A0).sum(1)).max(-1))
        ce_l.append(-(wt * np.log(wm + 1e-300)).sum(-1))
    c = lambda x: np.concatenate(x)  # noqa: E731
    return dict(rank=float(c(rk).mean()), regretW=float(c(rg).mean()), du0=float(c(du).mean()), ce=float(c(ce_l).mean()),
                n=int(len(idx)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dfl', required=True, help='comma-separated *_dfl.npz files (training courses)')
    ap.add_argument('--resid', default='mlp', choices=('mlp', 'gru'))
    ap.add_argument('--init', default='', help='residual checkpoint to start from (e.g. the v3 forecast-trained model)')
    ap.add_argument('--val_courses', type=int, default=3, help='number of courses (last, sorted) held out')
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--batch', type=int, default=16, help='decisions per iteration')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--w_dif', type=float, default=0.1)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--pre_full', type=int, default=1, help='1: also evaluate physics and the initial model on the full held-out set')
    ap.add_argument('--eval_max', type=int, default=160, help='held-out decisions for interim checks (fixed random subset)')
    ap.add_argument('--seed', type=int, default=2899)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='float32', choices=('float32', 'float64'))
    ap.add_argument('--m_train', type=int, default=0, help='candidates per decision in training (0: all stored)')
    ap.add_argument('--resume', type=int, default=0, help='1: continue from <out>.partial.pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = load_dfl(a.dfl.split(','))
    lam = D['lam']
    courses = sorted(set(D['course'].tolist())); val_c = courses[-a.val_courses:]
    tr = np.flatnonzero(~np.isin(D['course'], val_c)); va = np.flatnonzero(np.isin(D['course'], val_c))
    if a.init:
        R, sc = RK.load_resid(a.init)
    else:
        R, sc = (MLPResid() if a.resid == 'mlp' else GRUResid()).double(), 0.2
    global DT_
    DT_ = torch.float32 if a.dtype == 'float32' else torch.float64
    R = R.to(dev).to(DT_); R.train()
    model = ResidPhysics2(resid=R, scale=sc, device=dev, dtype=DT_, n_sub=5)
    base = ResidPhysics2(resid=None, device=dev, dtype=DT_, n_sub=5)
    opt = torch.optim.Adam(R.parameters(), lr=a.lr)
    t0 = time.time(); hist = []
    va_sub = np.sort(np.random.default_rng(7).choice(va, min(a.eval_max, len(va)), replace=False))
    part = Path(a.out + '.partial.pt'); stop_f = Path(a.out + '.STOP'); it0 = 0; evals = []
    if a.resume and part.exists():
        P = torch.load(part, map_location='cpu', weights_only=False)
        R.load_state_dict({k: v.to(dev).to(DT_) for k, v in P['state'].items()}); opt.load_state_dict(P['opt'])
        rng.bit_generator.state = P['np_rng']; torch.set_rng_state(P['torch_rng'].cpu())
        hist, evals, ev0, it0 = P['hist'], P['evals'], P['before'], P['it']
        t0 -= P['elapsed']
        print(f'  resumed at iteration {it0}', flush=True)
    else:
        ev0 = dict(physics_subset=evaluate(base, D, va_sub, dev, lam), init_subset=evaluate(model, D, va_sub, dev, lam))
        if a.pre_full:
            ev0.update(physics=evaluate(base, D, va, dev, lam), init=evaluate(model, D, va, dev, lam))
    print('  held-out before training:', json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ev0.items()}), flush=True)
    for it in range(it0, a.iters):
        if stop_f.exists():
            torch.save(dict(state={k: v.double() for k, v in R.state_dict().items()}, opt=opt.state_dict(),
                            np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), hist=hist, evals=evals,
                            before=ev0, it=it, elapsed=time.time() - t0, kind=a.resid, scale=sc, args=vars(a)), part)
            rep = dict(args=vars(a), n_train=int(len(tr)), n_val=int(len(va)), val_courses=val_c, before=ev0, evals=evals,
                       hist=hist, wall_s=time.time() - t0, stopped_early=True, stopped_at=it)
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text(json.dumps(rep, indent=1))
            print(f'  STOP file found at iteration {it}: state saved to {part}', flush=True)
            sys.exit(3)
        sub = rng.choice(tr, a.batch, replace=False)
        cand = None
        if a.m_train and a.m_train < D['A'].shape[1]:
            half = D['A'].shape[1] // 2; mh = a.m_train // 2
            cand = np.concatenate([np.sort(rng.choice(half, mh, replace=False)), half + np.sort(rng.choice(half, a.m_train - mh, replace=False))])
        Gm, _ = batch_values(model, D, sub, dev, grad=True, cand=cand)
        Gt = torch.as_tensor(D['Gtw'][sub] if cand is None else D['Gtw'][sub][:, cand], dtype=DT_, device=dev)
        L, ce, dif = losses(Gm, Gt, lam, a.w_dif)
        opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0); opt.step()
        if it % 50 == 0:
            hist.append((it, float(L), float(ce), float(dif)))
            print(f'  it{it} loss {float(L):.4f} ce {float(ce):.4f} dif {float(dif):.4f} ({time.time() - t0:.0f} s)', flush=True)
        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            last = it == a.iters - 1
            e = evaluate(model, D, va if last else va_sub, dev, lam); e['it'] = it + 1; e['subset'] = not last; evals.append(e)
            print('  held-out', json.dumps({k: round(v, 4) for k, v in e.items()}), flush=True)
            torch.save(dict(state={k: v.double() for k, v in R.state_dict().items()}, kind=a.resid, scale=sc, args=vars(a), it=it + 1),
                       a.out + f'.it{it + 1}.pt')
    rep = dict(args=vars(a), n_train=int(len(tr)), n_val=int(len(va)), val_courses=val_c, before=ev0, evals=evals, hist=hist,
               wall_s=time.time() - t0, note='final checkpoint = last iteration (no selection on held-out decisions)')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state={k: v.double() for k, v in R.state_dict().items()}, kind=a.resid, scale=sc, args=vars(a)), a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print('done', json.dumps(evals[-1]))


if __name__ == '__main__':
    main()
