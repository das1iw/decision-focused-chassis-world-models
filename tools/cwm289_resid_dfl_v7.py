"""Version 7 (2026-10-15 10:30): version 6 plus a FORECAST REGULARIZER. The v5/v6 decision-focused MLPs reach better
first actions and rankings by distorting the dynamics (1 s position error 101-104 mm vs 22 mm for plain physics, wheel
speed RMSE 8-9 vs 3.4 rad/s; 07 2026-10-15 10:22). Each iteration also draws --fc_batch windows of the causal sequence
dataset (training courses of tools/cwm289_resid_train_v3.py, 0.5 s open-loop rollouts with recorded commands) and adds
the v3 decision-aligned forecast loss (scales of the v3 MLP report) as one extra gradient group: the update is
    mean_b unit(g_b)  +  --w_fc * unit(g_forecast)
where g_b are the per-decision gradients of the decision loss (v5) and unit(.) normalizes to length 1. Both backward
passes are in the CUDA graph (hooks armed only for the decision pass). Everything else is version 6.

Decision-focused residual training, version 6 (2026-10-15 09:25): the method of version 5 (per-decision normalized
gradients of the decision loss, one backward pass per batch, MLP residual) with the forward+backward pass of a batch
captured once as a CUDA graph and replayed every iteration. Version 5 is launch-bound (about 150 000 small kernels per
iteration: 7.5 s per iteration alone on the GPU, 19 s on a shared GPU, 9 s on 6 CPU threads; 07 2026-10-15 09:12).
Batches are copied into static GPU buffers before each replay; the per-decision normalization and the Adam step run
eagerly after the replay. Loss, data, held-out checks (eager, no graph), checkpoints, STOP/resume and report format are
version 5's. Optional --w_reg adds the differentiable subset regret of the model's MPPI weights,
  R = sum_j p_tw,j G_tw,j - sum_j softmax(G_m / lambda)_j G_tw,j      (twin value lost by the model's weights),
to the loss (added 2026-10-15 10:2x after the first runs improved du0 and rank but not regretW; default 0 = v5 loss).
--check_graph N compares the graph's loss and per-decision gradients with an eager pass on one batch and
stops after N such comparisons.

    .venv-wm/Scripts/python.exe -B trfc_xcar/run_pinned284.py --cores 30-31 tools/cwm289_resid_dfl_v6.py --dfl ... --init ...
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
spec = importlib.util.spec_from_file_location('dfl5', ROOT / 'tools' / 'cwm289_resid_dfl_v5.py')
V5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V5)
from trfc_drift import mppi_cuda289_resid_v2 as RK  # noqa: E402
from trfc_drift.torch_objective289_v2 import physics_step  # noqa: E402
from trfc_drift.wm_physics2_resid289_v1 import MLPResid, ResidPhysics2  # noqa: E402
spec3 = importlib.util.spec_from_file_location('rt3', ROOT / 'tools' / 'cwm289_resid_train_v3.py')
RT3 = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(RT3)

KEYS_ROW = ('s0', 'h0', 'mu')                       # per decision, expanded to rows inside the graph
KEYS_EP = ('prev_a', 'st0', 'ts0', 'pose0')        # per decision (episode index = decision index)
KEYS_W = ('txy', 'tpsi', 'tsw', 'bref', 'vl', 's0p')


class StaticBatch:
    def __init__(self, D, B, M, dev, dt):
        self.D, self.B, self.M, self.dev, self.dt = D, B, M, dev, dt
        T, na = D['A'].shape[2], D['A'].shape[3]
        self.buf = {k: torch.zeros((B,) + D[k].shape[1:], dtype=dt, device=dev) for k in KEYS_ROW + KEYS_EP + KEYS_W}
        self.buf['A'] = torch.zeros(B, M, T, na, dtype=dt, device=dev)
        self.buf['Gt'] = torch.zeros(B, M, dtype=dt, device=dev)
        self.rows = torch.arange(B, device=dev).repeat_interleave(M)

    def load(self, idx, cand):
        D = self.D
        for k in KEYS_ROW + KEYS_EP + KEYS_W:
            self.buf[k].copy_(torch.as_tensor(D[k][idx], dtype=self.dt))
        A = D['A'][idx] if cand is None else D['A'][idx][:, cand]
        G = D['Gtw'][idx] if cand is None else D['Gtw'][idx][:, cand]
        self.buf['A'].copy_(torch.as_tensor(A, dtype=self.dt)); self.buf['Gt'].copy_(torch.as_tensor(G, dtype=self.dt))


def values_static(model, sb, eP):
    b, rows = sb.buf, sb.rows
    B, M, T, na = b['A'].shape
    W = {k: b[k] for k in KEYS_W}
    G, alive = V5.objective(physics_step(model), (b['h0'][rows], None), b['s0'][rows], b['A'].reshape(B * M, T, na),
                            b['mu'][rows], eP, W, rows, b['prev_a'], b['st0'], b['ts0'], b['pose0'], dtype=sb.dt)
    return G.reshape(B, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dfl', required=True)
    ap.add_argument('--init', default='')
    ap.add_argument('--val_courses', type=int, default=3)
    ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--m_train', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--w_dif', type=float, default=0.1)
    ap.add_argument('--w_reg', type=float, default=0.0, help='weight of the differentiable subset regret (0: v5 loss)')
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--eval_max', type=int, default=160)
    ap.add_argument('--eval_bs', type=int, default=32)
    ap.add_argument('--pre_full', type=int, default=0)
    ap.add_argument('--seed', type=int, default=2899)
    ap.add_argument('--check_graph', type=int, default=0)
    ap.add_argument('--seqdata', default='D:/TRFC_data/wm4_289/seqdata_v1/data.npz')
    ap.add_argument('--fc_scales', default='experiments/results/wm4_289/resid_v3_mlp.json', help='report holding the v3 loss scales')
    ap.add_argument('--fc_batch', type=int, default=64)
    ap.add_argument('--fc_H', type=int, default=25)
    ap.add_argument('--w_fc', type=float, default=1.0)
    ap.add_argument('--resume', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    dev = 'cuda'; dt = torch.float32; V5.DT_ = dt
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    D = V5.load_dfl(a.dfl.split(',')); lam = D['lam']; eP = D['eP']
    courses = sorted(set(D['course'].tolist())); val_c = courses[-a.val_courses:]
    tr = np.flatnonzero(~np.isin(D['course'], val_c)); va = np.flatnonzero(np.isin(D['course'], val_c))
    if a.init:
        R, sc = RK.load_resid(a.init)
    else:
        R, sc = MLPResid().double(), 0.2
    R = R.to(dev).to(dt); R.train()
    model = ResidPhysics2(resid=R, scale=sc, device=dev, dtype=dt, n_sub=5)
    base = ResidPhysics2(resid=None, device=dev, dtype=dt, n_sub=5)
    params = list(R.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    pdg = V5.PerDecisionGrads(R)
    half = D['A'].shape[1] // 2; mh = a.m_train // 2
    sb = StaticBatch(D, a.batch, a.m_train, dev, dt)
    SD = RT3.load(a.seqdata)
    cs_ = sorted(set(SD['course'].tolist())); fval = cs_[-4:]                          # trainer v3's default split
    f_tr = [e for e in range(len(SD['course'])) if SD['course'][e] not in fval]
    f_idx = RT3.windows(SD, f_tr, 0, a.fc_H, 3)
    RT3.SCALES = json.loads(Path(a.fc_scales).read_text())['loss_scales']
    fb0 = RT3.batch(SD, [f_idx[i] for i in range(a.fc_batch)], 0, a.fc_H, dev)
    fbuf = {k: v.to(dt).clone() for k, v in fb0.items()}

    def fc_load():
        sel = rng.choice(len(f_idx), a.fc_batch, replace=False)
        b = RT3.batch(SD, [f_idx[i] for i in sel], 0, a.fc_H, 'cpu')
        for k in fbuf:
            fbuf[k].copy_(b[k].to(dt))

    def draw():
        sub = rng.choice(tr, a.batch, replace=False)
        cand = np.concatenate([np.sort(rng.choice(half, mh, replace=False)), half + np.sort(rng.choice(half, a.m_train - mh, replace=False))])
        return sub, cand

    fc_grads = {}

    def fwd_bwd():
        pdg.arm(a.batch)
        Gm = values_static(model, sb, eP)
        L, ce, dif = V5.losses(Gm, sb.buf['Gt'], lam, a.w_dif)
        if a.w_reg > 0:
            Gt = sb.buf['Gt']; pt = torch.softmax(Gt / lam, -1); pm = torch.softmax(Gm / lam, -1)
            L = L + a.w_reg * ((pt * Gt).sum(-1) - (pm * Gt).sum(-1)).mean()
        torch.autograd.grad(L, params)
        acc_ = pdg.acc; pdg.B = None                                     # hooks off for the forecast pass
        Lf = RT3.loss_v3(RT3.rollout(model, fbuf, 0, a.fc_H), fbuf, 0, a.fc_H)
        gf = torch.autograd.grad(Lf, params)
        for q, g_ in zip(params, gf):
            fc_grads[q] = g_
        pdg.acc = acc_
        return L, ce, Lf

    # warm-up on a side stream, then capture
    sub, cand = draw(); sb.load(sub, cand)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            pdg.arm(a.batch); fwd_bwd(); pdg.grads()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    pdg.arm(a.batch)
    with torch.cuda.graph(graph):
        L_s, ce_s, dif_s = fwd_bwd()
    acc = pdg.acc; pdg.B = None; pdg.acc = {}          # keep the static accumulators, disarm the hooks for eager passes
    fc_static = dict(fc_grads)                          # the graph's forecast-gradient tensors
    torch.cuda.synchronize()

    def per_decision():
        return {q: acc[q] for q in params}

    if a.check_graph:
        res = []
        for _ in range(a.check_graph):
            sub, cand = draw(); sb.load(sub, cand); fc_load(); graph.replay(); torch.cuda.synchronize()
            pg = {q: t.clone() for q, t in per_decision().items()}; Lg = float(L_s)
            pdg.arm(a.batch); Le, _, _ = fwd_bwd(); pe = pdg.grads(); pdg.B = None
            res.append(dict(loss_graph=Lg, loss_eager=float(Le),
                            grad_rel=max(float((pg[q] - pe[q]).abs().max() / (pe[q].abs().max() + 1e-30)) for q in params)))
            print('check_graph', json.dumps(res[-1]), flush=True)
        t0 = time.time()
        for _ in range(10):
            sub, cand = draw(); sb.load(sub, cand); fc_load(); graph.replay()
        torch.cuda.synchronize(); tg = (time.time() - t0) / 10
        t0 = time.time()
        for _ in range(2):
            pdg.arm(a.batch); fwd_bwd(); pdg.grads(); pdg.B = None
        torch.cuda.synchronize(); te = (time.time() - t0) / 2
        out = dict(args=vars(a), checks=res, s_per_iter_graph=tg, s_per_iter_eager=te)
        print(json.dumps(out), flush=True)
        Path(a.report).parent.mkdir(parents=True, exist_ok=True); Path(a.report).write_text(json.dumps(out, indent=1))
        sys.exit(0)

    t0 = time.time(); hist = []; evals = []
    va_sub = np.sort(np.random.default_rng(7).choice(va, min(a.eval_max, len(va)), replace=False))
    part = Path(a.out + '.partial.pt'); stop_f = Path(a.out + '.STOP'); it0 = 0
    if a.resume and part.exists():
        P = torch.load(part, map_location='cpu', weights_only=False)
        with torch.no_grad():
            for q, (k, v) in zip(params, P['state'].items()):
                q.copy_(v.to(dev).to(dt))                # in place: the captured graph keeps the parameter storage
        opt.load_state_dict(P['opt']); rng.bit_generator.state = P['np_rng']
        hist, evals, ev0, it0 = P['hist'], P['evals'], P['before'], P['it']; t0 -= P['elapsed']
        print(f'  resumed at iteration {it0}', flush=True)
    else:
        ev0 = dict(physics_subset=V5.evaluate(base, D, va_sub, dev, lam, bs=a.eval_bs), init_subset=V5.evaluate(model, D, va_sub, dev, lam, bs=a.eval_bs))
        if a.pre_full:
            ev0.update(physics=V5.evaluate(base, D, va, dev, lam, bs=a.eval_bs), init=V5.evaluate(model, D, va, dev, lam, bs=a.eval_bs))
    print('  held-out before training:', json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ev0.items()}), flush=True)
    for it in range(it0, a.iters):
        if stop_f.exists():
            torch.save(dict(state={k: v.double() for k, v in R.state_dict().items()}, opt=opt.state_dict(), np_rng=rng.bit_generator.state,
                            hist=hist, evals=evals, before=ev0, it=it, elapsed=time.time() - t0, kind='mlp', scale=sc, args=vars(a)), part)
            rep = dict(args=vars(a), n_train=int(len(tr)), n_val=int(len(va)), val_courses=val_c, before=ev0, evals=evals, hist=hist,
                       wall_s=time.time() - t0, stopped_early=True, stopped_at=it)
            Path(a.report).parent.mkdir(parents=True, exist_ok=True); Path(a.report).write_text(json.dumps(rep, indent=1))
            print(f'  STOP file found at iteration {it}: state saved to {part}', flush=True)
            sys.exit(3)
        sub, cand = draw(); sb.load(sub, cand); fc_load(); graph.replay()
        per = per_decision()
        nb = torch.sqrt(sum((per[q].double() ** 2).reshape(a.batch, -1).sum(1) for q in params))
        good = torch.isfinite(nb) & (nb > 0); ng = max(1, int(good.sum()))
        w = torch.where(good, 1.0 / nb.clamp(min=1e-30), torch.zeros_like(nb))
        opt.zero_grad(set_to_none=False)
        nf = torch.sqrt(sum((fc_static[q].double() ** 2).sum() for q in params))
        wf = a.w_fc / float(nf) if bool(torch.isfinite(nf)) and float(nf) > 0 else 0.0
        for q in params:
            q.grad = (per[q] * w.to(per[q].dtype).reshape(-1, *([1] * (per[q].dim() - 1)))).sum(0) / ng + wf * fc_static[q]
        opt.step()
        if it % 50 == 0:
            hist.append((it, float(L_s), float(ce_s), float(dif_s)))
            gn_ = nb[good].cpu().numpy()
            print(f'  it{it} loss {float(L_s):.4f} ce {float(ce_s):.4f} forecast {float(dif_s):.4f} per-decision |g| median {np.median(gn_):.3g} '
                  f'max {np.max(gn_):.3g} ({time.time() - t0:.0f} s)', flush=True)
        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            last = it == a.iters - 1
            e = V5.evaluate(model, D, va if last else va_sub, dev, lam, bs=a.eval_bs); e['it'] = it + 1; e['subset'] = not last; evals.append(e)
            print('  held-out', json.dumps({k: round(v, 4) for k, v in e.items()}), flush=True)
            torch.save(dict(state={k: v.double() for k, v in R.state_dict().items()}, kind='mlp', scale=sc, args=vars(a), it=it + 1),
                       a.out + f'.it{it + 1}.pt')
    rep = dict(args=vars(a), n_train=int(len(tr)), n_val=int(len(va)), val_courses=val_c, before=ev0, evals=evals, hist=hist,
               wall_s=time.time() - t0, note='final checkpoint = last iteration (no selection on held-out decisions)')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state={k: v.double() for k, v in R.state_dict().items()}, kind='mlp', scale=sc, args=vars(a)), a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True); Path(a.report).write_text(json.dumps(rep, indent=1))
    print('done', json.dumps(evals[-1]))


if __name__ == '__main__':
    main()
