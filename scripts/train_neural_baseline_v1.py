"""Train neural baseline."""
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
SCALE = torch.tensor([20.0, 10.0, 1.0, 60.0, 60.0, 60.0, 60.0, 500.0, 500.0, 500.0, 500.0])
LOSS_SC = torch.tensor([1.0, 1.0, 0.5, 5.0, 5.0, 5.0, 5.0, 200.0, 200.0, 200.0, 200.0])
EVAL_SC = np.array([1.0, 1.0, 0.5, 5.0, 5.0, 5.0, 5.0])


def _train_mod():
    spec = importlib.util.spec_from_file_location('rt', ROOT / 'scripts' / 'train_forecast_v1.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class PureGRU(torch.nn.Module):
    def __init__(self, hidden=128, width=256):
        super().__init__()
        nin = 11 + 8 + 1
        self.cell = torch.nn.GRUCell(nin, hidden)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden + nin, width), torch.nn.SiLU(), torch.nn.Linear(width, width),
                                        torch.nn.SiLU(), torch.nn.Linear(width, 11))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01); self.head[-1].bias.zero_()
        self.hidden = hidden

    def x(self, s, u, mu):
        uu = torch.cat([u[:, :4], u[:, 4:] / 500.0], 1)
        return torch.cat([s / SCALE.to(s), uu, mu[:, None]], 1)

    def step(self, s, u, mu, z):
        x = self.x(s, u, mu)
        z = self.cell(x, z if z is not None else torch.zeros(s.shape[0], self.hidden, dtype=s.dtype, device=s.device))
        ds = self.head(torch.cat([z, x], 1)) * 0.1 * SCALE.to(s)
        return s + ds, z


def rollout(model, b, W, H):
    z = None
    for j in range(W):
        _, z = model.step(b['sig'][:, j, :11], b['u'][:, j], b['mu'][:, j], z)
    s = b['sig'][:, W, :11]; out = []
    for j in range(H):
        s, z = model.step(s, b['u'][:, W + j], b['mu'][:, W + j], z)
        out.append(s)
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--val_courses', default='')
    ap.add_argument('--warm', type=int, default=10)
    ap.add_argument('--iters', type=int, nargs=3, default=[600, 600, 800])
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--seed', type=int, default=289)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    tm = _train_mod()
    torch.manual_seed(a.seed); dev = 'cpu'
    D = tm.load(a.data)
    courses = sorted(set(D['course'].tolist()))
    val_c = [int(c) for c in a.val_courses.split(',')] if a.val_courses else courses[-4:]
    tr_e = [e for e in range(len(D['course'])) if D['course'][e] not in val_c]
    va_e = [e for e in range(len(D['course'])) if D['course'][e] in val_c]
    model = PureGRU(a.hidden, a.width)
    n_par = sum(p.numel() for p in model.parameters())
    W = a.warm
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist = []; t0 = time.time(); rng = np.random.default_rng(a.seed)

    def fbatch(idx, H):
        b = tm.batch(D, idx, W, H, dev)
        return {k: v.float() for k, v in b.items()}

    for H, n_it in zip((5, 15, 25), a.iters):
        idx = tm.windows(D, tr_e, W, H, 3)
        for it in range(n_it):
            sel = rng.choice(len(idx), min(a.batch, len(idx)), replace=False)
            b = fbatch([idx[i] for i in sel], H)
            pred = rollout(model, b, W, H)
            tg = b['sig'][:, W + 1:W + H + 1, :11]
            l = (((pred - tg) / LOSS_SC) ** 2).mean()
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if it % 100 == 0 or it == n_it - 1:
                hist.append((H, it, float(l.detach()))); print(f'  H{H} it{it} loss {float(l.detach()):.5f} ({time.time() - t0:.0f} s)', flush=True)

    @torch.no_grad()
    def evaluate(max_n=3000):
        rep = {}
        for Hs, H in (('0.1s', 5), ('0.2s', 10), ('0.5s', 25), ('1.0s', 50)):
            idx = tm.windows(D, va_e, W, H, 10, max_n=max_n)
            errs = []
            for i in range(0, len(idx), 512):
                b = fbatch(idx[i:i + 512], H)
                p = rollout(model, b, W, H)
                errs.append((p[:, -1, :7] - b['sig'][:, W + H, :7]).double().numpy())
            E = np.concatenate(errs)
            rep[Hs] = dict(n=len(E), rmse_vx=float(np.sqrt((E[:, 0] ** 2).mean())), rmse_vy=float(np.sqrt((E[:, 1] ** 2).mean())),
                           rmse_r=float(np.sqrt((E[:, 2] ** 2).mean())), rmse_omega=float(np.sqrt((E[:, 3:7] ** 2).mean())),
                           score=float(((E / EVAL_SC) ** 2).mean()))
        return rep

    rep = dict(args=vars(a), n_params=n_par, val_courses=val_c, n_train_eps=len(tr_e), n_val_eps=len(va_e), hist=hist,
               wall_s=time.time() - t0, val=evaluate())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), kind='pure_gru', hidden=a.hidden, width=a.width, args=vars(a)), a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    for Hs in rep['val']:
        print(Hs, json.dumps({k: round(v, 4) for k, v in rep['val'][Hs].items()}))


if __name__ == '__main__':
    main()
