"""Learned chassis world models for the DDEV drift programme (research/105 phase D): hybrid and pure-neural.

Hybrid: s_{k+1} = physics2(s_k, h_k, u_{k+1}, mu) + D_theta(z_k, g_k), with g a GRU context that runs through the
teacher-forced warm-up and then the open-loop rollout. z holds the normalized state, input, mu, the physics
increment and the physics hidden state (slip lags, load filter, forces). The last layer starts at zero, so an
untrained hybrid equals physics2.
Neural: the same network with the physics step replaced by persistence (s_p = s, no physics features).

Training: multi-step open-loop rollouts (W = 10 teacher-forced warm-up steps, H = 25 open-loop steps) through the
differentiable physics (10 substeps of 2 ms), loss = channel-normalized MSE over the rollout plus 0.1 x the warm-up
one-step loss. Data split by actor (whole actor streams stay together): train actors 0-5, validation 6-7,
development test 8-9. CPU only (the GPU belongs to the protocol-107 learners).

Usage: python -m trfc_drift.wm_hybrid train --kind hybrid --out D:\\TRFC_data\\drift_wm\\hybrid_dev_s1 --seed 1
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from trfc_drift import carsim_dll as cs
from trfc_drift import wm_data as W

RUNS_DEV = (r"D:\TRFC_data\drift_runs\pilot6_drift_4wstv_estenv_s1", r"D:\TRFC_data\drift_runs\pilot6_drift_2wsrwd_none_s1")
SPLIT = dict(train=range(0, 6), val=range(6, 8), devtest=range(8, 10))
S_SCALE = torch.tensor([20.0, 5.0, 1.0, 50.0, 50.0, 50.0, 50.0, 1000.0, 1000.0, 1000.0, 1000.0])
D_SCALE = torch.tensor([0.1, 0.1, 0.05, 5.0, 5.0, 5.0, 5.0, 100.0, 100.0, 100.0, 100.0])
U_SCALE = torch.tensor([0.5, 0.5, 0.2, 0.2, 2000.0, 2000.0, 2000.0, 2000.0])
H_SCALE = torch.tensor([1.0] * 4 + [1.0] * 4 + [10.0, 10.0] + [3000.0] * 8)
LOSS_W = torch.tensor([1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1])


# ---------------- data ----------------
def load_runs(runs, actors, max_files=None):
    """Concatenate the state shards of the given actors. Returns wm_data-style dict plus mu_hat per row."""
    D, EP, MH, n = [], [], [], 0
    for run in runs:
        eps_meta = {}
        ej = Path(run) / "episodes.jsonl"
        if ej.exists():
            for line in ej.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eps_meta[(int(r["actor"]), int(r["ep"]))] = float(r.get("mu_hat", np.nan))
        files = sorted(glob.glob(str(Path(run) / "states" / "*.npz")))
        files = [f for f in files if int(re.search(r"actor(\d+)_", Path(f).name).group(1)) in actors]
        for f in files[:max_files]:
            actor = int(re.search(r"actor(\d+)_", Path(f).name).group(1))
            z = np.load(f)
            for ep, mu, ln in z["episodes"]:
                EP.append((n, int(ln), float(mu)))
                MH.append(np.full(int(ln), eps_meta.get((actor, int(ep)), np.nan)))
                n += int(ln)
            D.append(z["data"].astype(np.float64))
    data = np.concatenate(D)
    E = data[:, :cs.N_EXP]
    return dict(S=W.row_state(E), U=data[:, cs.N_EXP:cs.N_EXP + 8], mu=data[:, cs.N_EXP + 8], extra=W.row_extra(E),
                episodes=np.array(EP, dtype=np.float64), mu_hat=np.concatenate(MH))


def starts(d, Wm, H):
    idx = [np.arange(int(g) + Wm, int(g) + int(ln) - H) for g, ln, _ in d["episodes"] if ln > Wm + H]
    return np.concatenate(idx)


class Batcher:
    def __init__(self, d, Wm, H, dtype=torch.float32):
        self.Wm, self.H = Wm, H
        self.S = torch.as_tensor(d["S"], dtype=dtype)
        self.U = torch.as_tensor(d["U"], dtype=dtype)
        self.MU = torch.as_tensor(d["mu"], dtype=dtype)
        self.AXY = torch.as_tensor(np.column_stack([d["extra"]["ax"], d["extra"]["ay"]]), dtype=dtype)
        self.k0 = starts(d, Wm, H)

    def get(self, k0):
        kk = torch.as_tensor(k0)
        hist = kk[:, None] + torch.arange(-self.Wm, 1)[None, :]
        fut = kk[:, None] + torch.arange(1, self.H + 1)[None, :]
        return (self.S[hist], self.U[hist[:, 1:]], self.U[fut], self.MU[kk], self.AXY[kk - self.Wm], self.S[fut])


# ---------------- model ----------------
class ChassisWM(nn.Module):
    def __init__(self, kind="hybrid", hidden=128, width=256, n_sub=10):
        super().__init__()
        self.kind = kind
        self.phys = None
        if kind == "hybrid":
            from trfc_drift.wm_physics2 import DDEVPhysics2
            self.phys = DDEVPhysics2("cpu", torch.float32, n_sub=n_sub)
        nh = 18 if kind == "hybrid" else 0
        self.in_dim = 11 + 8 + 1 + (11 if kind == "hybrid" else 0) + nh
        self.gru = nn.GRUCell(self.in_dim, hidden)
        self.head = nn.Sequential(nn.Linear(self.in_dim + hidden, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, 11))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.hidden = hidden

    def _feat(self, s, u, mu, sp, hp):
        parts = [s / S_SCALE, u / U_SCALE, mu[:, None]]
        if self.phys is not None:
            parts += [(sp - s) / D_SCALE, torch.clamp(hp / H_SCALE, -5, 5)]
        return torch.cat(parts, 1)

    def step(self, s, hp, g, u, mu):
        if self.phys is not None:
            sp, hp = self.phys.step(s, hp, u, mu)
        else:
            sp = s
        z = self._feat(s, u, mu, sp, hp)
        g = self.gru(z, g)
        return sp + self.head(torch.cat([z, g], 1)) * D_SCALE, hp, g

    def warmup(self, S_hist, U_hist, mu, axy0):
        B = S_hist.shape[0]
        hp = self.phys.init_hidden(S_hist[:, 0], U_hist[:, 0], mu, axy0[:, 0], axy0[:, 1]) if self.phys else None
        g = torch.zeros(B, self.hidden)
        preds = []
        for j in range(U_hist.shape[1]):
            sj, hp, g = self.step(S_hist[:, j], hp, g, U_hist[:, j], mu)
            preds.append(sj)
        return hp, g, torch.stack(preds, 1)

    def rollout(self, S_hist, U_hist, U_fut, mu, axy0, return_warm=False):
        hp, g, warm = self.warmup(S_hist, U_hist, mu, axy0)
        s = S_hist[:, -1]
        out, frc = [], []
        for k in range(U_fut.shape[1]):
            s, hp, g = self.step(s, hp, g, U_fut[:, k], mu)
            out.append(s)
            if self.phys is not None and not torch.is_grad_enabled():
                frc.append(torch.stack([hp[:, 10:14], hp[:, 14:18], self.phys.fz(hp[:, 8], hp[:, 9])], 1))
        self.last_forces = torch.stack(frc, 1) if frc else None
        out = torch.stack(out, 1)
        return (out, warm) if return_warm else out


class EvalAdapter:
    """wm_rollout_eval interface: rollout(S_hist, U_hist, U_fut, mu, extra0) with extra0 = [ax, ay at k0, at k0-W]."""

    def __init__(self, model, name):
        self.m, self.name = model, name

    def rollout(self, S_hist, U_hist, U_fut, mu, extra0):
        f = lambda x: x.float()  # noqa: E731
        with torch.no_grad():
            out = self.m.rollout(f(S_hist), f(U_hist), f(U_fut), f(mu), f(extra0[:, 2:4])).double()
        self.last_forces = None if self.m.last_forces is None else self.m.last_forces.double()
        return out


def load_model(name, device="cpu", dtype=torch.float64):
    """name = 'ckpt:<path to .pt>' """
    path = Path(name.split(":", 1)[1])
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ChassisWM(**ck["cfg"])
    m.load_state_dict(ck["state"])
    m.eval()
    return EvalAdapter(m, f"{ck['cfg']['kind']}:{path.parent.name}")


# ---------------- training ----------------
def rollout_loss(model, batch, sig):
    S_hist, U_hist, U_fut, mu, axy0, S_fut = batch
    pred, warm = model.rollout(S_hist, U_hist, U_fut, mu, axy0, return_warm=True)
    lw = LOSS_W / sig ** 2
    main = (((pred - S_fut) ** 2) * lw).mean()
    wl = (((warm - S_hist[:, 1:]) ** 2) * lw).mean()
    return main + 0.1 * wl, main


def evaluate_val(model, bv, sig, rng, n=1000):
    k0 = rng.choice(bv.k0, min(n, len(bv.k0)), replace=False)
    with torch.no_grad():
        S_hist, U_hist, U_fut, mu, axy0, S_fut = bv.get(k0)
        pred = model.rollout(S_hist, U_hist, U_fut, mu, axy0)
    e = (pred - S_fut)
    out = {}
    for h in (10, 25, 50):
        if h <= pred.shape[1]:
            rm = torch.sqrt((e[:, h - 1] ** 2).mean(0))
            out[h] = float(((rm[:3] / sig[:3]).sum() + (rm[3:7] / sig[3:7]).mean()) / 4)
    return out


def train(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    dtr = load_runs(RUNS_DEV, set(SPLIT["train"]), a.max_files)
    dva = load_runs(RUNS_DEV, set(SPLIT["val"]), a.max_files)
    btr, bva = Batcher(dtr, 10, a.H), Batcher(dva, 10, 50)
    sig = torch.as_tensor(np.std(dtr["S"], 0), dtype=torch.float32)
    sig[3:7] = float(np.std(dtr["S"][:, 3:7]))
    sig[7:11] = float(np.std(dtr["S"][:, 7:11]))
    cfg = dict(kind=a.kind, hidden=a.hidden, width=a.width, n_sub=a.n_sub)
    model = ChassisWM(**cfg)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=a.lr * 0.01)
    log = open(out / "train_log.jsonl", "a")
    best, hist = None, []
    info = dict(cfg=cfg, n_params=sum(p.numel() for p in model.parameters()), train_rows=int(len(dtr["S"])),
                val_rows=int(len(dva["S"])), train_windows=int(len(btr.k0)), val_windows=int(len(bva.k0)),
                sig=sig.tolist(), args=vars(a))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    v0 = evaluate_val(model, bva, sig, np.random.default_rng(123))
    print(json.dumps(dict(it=0, val=v0, **{k: info[k] for k in ("n_params", "train_windows", "val_windows")})), flush=True)
    for it in range(1, a.iters + 1):
        k0 = rng.choice(btr.k0, a.batch, replace=False)
        loss, main = rollout_loss(model, btr.get(k0), sig)
        if not torch.isfinite(loss):
            print(json.dumps(dict(it=it, error="nonfinite loss")), flush=True)
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % a.eval_every == 0 or it == a.iters:
            v = evaluate_val(model, bva, sig, np.random.default_rng(123))
            rec = dict(it=it, loss=float(loss.detach()), main=float(main.detach()), grad=float(gn), val=v, wall=time.perf_counter() - t0)
            hist.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if best is None or v[25] < best[0]:
                best = (v[25], it)
                torch.save(dict(cfg=cfg, state=model.state_dict(), it=it, val=v), out / "best.pt")
    torch.save(dict(cfg=cfg, state=model.state_dict(), it=a.iters, val=hist[-1]["val"]), out / "final.pt")
    summ = dict(info, best_it=best[1], best_val_h25=best[0], val_untrained=v0, wall_s=time.perf_counter() - t0)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(dict(done=True, best_it=best[1], best_val_h25=best[0])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--kind", choices=("hybrid", "neural"), default="hybrid")
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--iters", type=int, default=3000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--H", type=int, default=25)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--n_sub", type=int, default=10)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--eval_every", type=int, default=250)
    t.add_argument("--max_files", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)


if __name__ == "__main__":
    main()
