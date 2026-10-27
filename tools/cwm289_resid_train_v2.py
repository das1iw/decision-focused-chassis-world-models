"""Version 2 of tools/cwm289_resid_train_v1.py (kept): identical training except `--device` (cpu or cuda; float64 either
way) and a graceful pause: the file <out>.STOP is checked every 25 iterations; on it the model, optimizer, both RNG
states, stage and iteration are saved to <out>.partial.pt and the process exits (code 3) without a report.
`--resume` continues from that file with the identical batch sequence (GPU reductions may differ in the last bits).

Train bounded tire-force residual ports of the planar drifting CWM against the 44-state twin (research/289 step 2).

    .venv-wm/Scripts/python.exe -B tools/run_pinned_low289.py --cores 28-31 tools/cwm289_resid_train_v1.py \
        --data D:/TRFC_data/wm4_289/seqdata_v1/data.npz --resid gru --out D:/TRFC_data/wm4_289/resid_v1/gru.pt \
        --report experiments/results/wm4_289/resid_v1_gru.json

Data: tools/cwm289_seqdata_v1.py (planar planner driving the twin; per step plant signals before the step, applied
commands, the planner's observer hidden state, true friction). Split by whole course (--val_courses, declared before
training; default the last 4 of the collection). Windows start at live steps whose next H steps are live.
Model: trfc_drift/wm_physics2_resid289_v1.ResidPhysics2 (planar physics v2, n_sub 5) with residual none | mlp | gru at
the tire-force ports. GRU memory is built causally over W warm-up steps (teacher-forced: state reset to the observed
signals and the recorded observer hidden state each step, memory carried), then the H-step forecast is open loop.
Loss: mean over the horizon of squared errors of vx, vy (1 m/s), r (0.5 rad/s), wheel speeds (5 rad/s); curriculum
H = 5, 15, 25 steps (0.1-0.5 s). Evaluation on the validation courses: 0.1 / 0.2 / 0.5 / 1.0 s forecast errors per
channel, overall and by stratum (pre/post drop, |beta| > 15 deg, friction), for the trained model and for no residual.
Development; the planar model is otherwise unchanged; the true friction is the friction input (the planner receives an
estimate; the model's job here is the dynamics given the friction).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trfc_drift.wm_physics2_resid289_v1 import GRUResid, MLPResid, ResidPhysics2  # noqa: E402

SC = torch.tensor([1.0, 1.0, 0.5, 5.0, 5.0, 5.0, 5.0], dtype=torch.float64)


def load(path):
    z = np.load(path)
    tags = sorted({k.split('_')[0] for k in z.files})
    D = dict(sig=[], u=[], h=[], mu=[], live=[], course=[], t_drop=[], k_drop=[])
    for tg in tags:
        D['sig'].append(z[f'{tg}_sig']); D['u'].append(z[f'{tg}_u']); D['h'].append(z[f'{tg}_h']); D['mu'].append(z[f'{tg}_mu'])
        D['live'].append(z[f'{tg}_live']); D['course'].append(z[f'{tg}_course']); D['t_drop'].append(z[f'{tg}_t_drop'])
    n = min(x.shape[1] for x in D['sig'])
    out = {k: np.concatenate([x[:, :n] for x in v]) if k in ('sig', 'u', 'h', 'mu', 'live') else np.concatenate(v) for k, v in D.items()
           if v}
    return out


def windows(D, eps, W, H, stride, rng=None, max_n=None):
    idx = []
    live = D['live']
    for e in eps:
        T = live.shape[1]
        for k in range(W, T - H - 1, stride):
            if live[e, k - W:k + H + 1].min() > 0.5:
                idx.append((e, k))
    if max_n and len(idx) > max_n:
        sel = (rng or np.random.default_rng(0)).choice(len(idx), max_n, replace=False); idx = [idx[i] for i in sorted(sel)]
    return idx


def batch(D, idx, W, H, dev):
    e = np.array([i[0] for i in idx]); k = np.array([i[1] for i in idx])
    rngk = np.arange(-W, H + 1)
    kk = k[:, None] + rngk[None]
    f = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)  # noqa: E731
    return dict(sig=f(D['sig'][e[:, None], kk]), u=f(D['u'][e[:, None], kk]), h=f(D['h'][e[:, None], kk]),
                mu=f(D['mu'][e[:, None], kk]))


def rollout(model, b, W, H):
    """Warm-up W teacher-forced steps (memory only), then H open-loop steps. Returns predictions [B, H, 7]."""
    z = None
    for j in range(W):
        _, _, z = model.step(b['sig'][:, j, :11], b['h'][:, j], b['u'][:, j], b['mu'][:, j], z=z)
    s = b['sig'][:, W, :11]; hh = b['h'][:, W]; out = []
    for j in range(H):
        s, hh, z = model.step(s, hh, b['u'][:, W + j], b['mu'][:, W + j], z=z)
        out.append(s[:, :7])
    return torch.stack(out, 1)


def loss_fn(pred, b, W, H):
    tg = b['sig'][:, W + 1:W + H + 1, :7]
    return (((pred - tg) / SC.to(pred)) ** 2).mean()


@torch.no_grad()
def evaluate(model, D, eps, W, dev, max_n=3000):
    rep = {}
    for Hs, H in (('0.1s', 5), ('0.2s', 10), ('0.5s', 25), ('1.0s', 50)):
        idx = windows(D, eps, W, H, 10, max_n=max_n)
        if not idx:
            continue
        errs = []
        for i in range(0, len(idx), 512):
            b = batch(D, idx[i:i + 512], W, H, dev)
            p = rollout(model, b, W, H)
            errs.append((p[:, -1] - b['sig'][:, W + H, :7]).cpu().numpy())
        E = np.concatenate(errs)
        rep[Hs] = dict(n=len(E), rmse_vx=float(np.sqrt((E[:, 0] ** 2).mean())), rmse_vy=float(np.sqrt((E[:, 1] ** 2).mean())),
                       rmse_r=float(np.sqrt((E[:, 2] ** 2).mean())), rmse_omega=float(np.sqrt((E[:, 3:7] ** 2).mean())),
                       score=float(((E / SC.numpy()) ** 2).mean()))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--resid', default='gru', choices=('none', 'mlp', 'gru'))
    ap.add_argument('--val_courses', default='')
    ap.add_argument('--warm', type=int, default=10)
    ap.add_argument('--iters', type=int, nargs=3, default=[600, 600, 800])
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--scale', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=289)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--resume', action='store_true', help='continue from <out>.partial.pt if it exists')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); dev = a.device
    D = load(a.data)
    courses = sorted(set(D['course'].tolist()))
    val_c = [int(c) for c in a.val_courses.split(',')] if a.val_courses else courses[-4:]
    tr_e = [e for e in range(len(D['course'])) if D['course'][e] not in val_c]
    va_e = [e for e in range(len(D['course'])) if D['course'][e] in val_c]
    R = None if a.resid == 'none' else (MLPResid() if a.resid == 'mlp' else GRUResid()).double().to(dev)
    model = ResidPhysics2(resid=R, scale=a.scale, device=dev, dtype=torch.float64, n_sub=5)
    W = a.warm if a.resid == 'gru' else 0
    hist = []; t0 = time.time(); rng = np.random.default_rng(a.seed)
    if R is not None:
        opt = torch.optim.Adam(R.parameters(), lr=a.lr)
        part = Path(a.out + '.partial.pt'); stop = Path(a.out + '.STOP')
        st0, it0 = 0, 0
        if a.resume and part.exists():
            ck = torch.load(part, map_location=dev, weights_only=False)
            R.load_state_dict(ck['state']); opt.load_state_dict(ck['opt']); rng.bit_generator.state = ck['np_rng']
            torch.set_rng_state(ck['torch_rng'].cpu()); hist = ck['hist']; st0, it0 = ck['stage'], ck['it'] + 1
            t0 -= ck['elapsed']
            print(f'  resumed at stage {st0} iteration {it0} ({ck["elapsed"]:.0f} s done before)', flush=True)
        for si, (H, n_it) in enumerate(zip((5, 15, 25), a.iters)):
            if si < st0:
                continue
            idx = windows(D, tr_e, W, H, 3)
            for it in range(it0 if si == st0 else 0, n_it):
                sel = rng.choice(len(idx), min(a.batch, len(idx)), replace=False)
                b = batch(D, [idx[i] for i in sel], W, H, dev)
                l = loss_fn(rollout(model, b, W, H), b, W, H)
                opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0); opt.step()
                if it % 100 == 0 or it == n_it - 1:
                    hist.append((H, it, float(l))); print(f'  H{H} it{it} loss {float(l):.5f} ({time.time() - t0:.0f} s)', flush=True)
                if it % 25 == 24 and stop.exists():
                    torch.save(dict(state=R.state_dict(), opt=opt.state_dict(), np_rng=rng.bit_generator.state,
                                    torch_rng=torch.get_rng_state(), hist=hist, stage=si, it=it, elapsed=time.time() - t0,
                                    args=vars(a)), part)
                    print(f'  STOP file found: paused at stage {si} iteration {it}; state saved to {part}', flush=True)
                    sys.exit(3)
    base = ResidPhysics2(resid=None, device=dev, dtype=torch.float64, n_sub=5)
    rep = dict(args=vars(a), val_courses=val_c, n_train_eps=len(tr_e), n_val_eps=len(va_e), hist=hist, wall_s=time.time() - t0,
               val=evaluate(model, D, va_e, W, dev), val_no_residual=evaluate(base, D, va_e, W, dev))
    if R is not None:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=R.state_dict(), kind=a.resid, scale=a.scale, args=vars(a)), a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    for Hs in rep['val']:
        print(Hs, 'model', json.dumps({k: round(v, 4) for k, v in rep['val'][Hs].items()}), '| no residual',
              json.dumps({k: round(v, 4) for k, v in rep['val_no_residual'][Hs].items()}))


if __name__ == '__main__':
    main()
