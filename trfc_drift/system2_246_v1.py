"""System 2 for the world-model-trained drift/race policy (research/246, development; author decision 2026-10-06 09:30).

System 1 is the RL policy (50 Hz). System 2 consults the 44-state chassis world model every `period` control steps and
acts only through System 1's friction input mu_hat:
(a) friction inference: replay the last window of applied actions from the state at the window start under friction
    hypotheses mu_probe * f (f in FACTORS) in an imagination copy of the environment, score the predicted vx, vy, r and
    wheel spins against the observed ones (Gaussian, per-channel sigma), and update a discrete posterior with forgetting
    and a floor (the protocol-137 filter, trfc_drift/wm_mu_filter.py, here on the 44-state model);
(b) look-ahead shield: from the current state, roll the world model `horizon` steps ahead with System 1 in the loop for
    candidate inputs mu_hat_c = mu_post * c (c in CANDIDATES, descending) in a world whose friction is the posterior's
    low quantile; feed System 1 the highest candidate whose rollout does not crash (all crash: the lowest).
Imagination copies are FastDriftEnv246 instances on the same courses; state transfer copies every kernel and environment
buffer (tested bit-exact). Inside the world model System 2 starts from the exact state; in CarSim this needs an
initializer (oracle internal states first; a causal initializer is open).
"""
from __future__ import annotations

import math

import numpy as np
import torch

KERNEL = ('y', 'fp', 'cache', 'kp', 'steer', 'motor', 'brake', 'external', 'mu', 'bad', 'lastbuf')
ENVB = ('tid', 'idx', 's_tr', 's_prog', 't', 'slow_t', 'laps', 'prev_a', 'steer_state', 'torque_state', 'mu', 'mu_hat', 'e_lat', 'e_psi',
        'zone_n', 'zone_ok', 'beta_ref_now')
FACTORS = np.round(np.arange(0.60, 1.4001, 0.05), 3)               # 17 hypotheses relative to mu_probe
CANDIDATES = (1.0, 0.85, 0.7, 0.55)
# 0.2 s replay noise per channel vx, vy, r, omega x4 (protocol-137 SIGMA0, physics v2 vs CarSim; conservative here)
SIGMA0 = np.array([0.0094, 0.0363, 0.0145, 1.2352, 0.8470, 2.3800, 1.7767])


def snapshot(env, rows=None):
    """Clones of every kernel, environment and signal buffer (rows subset if given)."""
    sel = (lambda x: x.clone()) if rows is None else (lambda x: x[rows].clone())
    snap = {('k', n): sel(getattr(env.kernel, n)) for n in KERNEL}
    snap.update({('e', n): sel(getattr(env, n)) for n in ENVB})
    snap.update({('s', n): sel(env.sig[n]) for n in env.SIG})
    return snap


def load(env, snap, dst_rows, src_index):
    """Write snapshot entries src_index into env rows dst_rows (in place: CUDA-graph buffers stay valid)."""
    for (kind, n), v in snap.items():
        tgt = getattr(env.kernel, n) if kind == 'k' else (getattr(env, n) if kind == 'e' else env.sig[n])
        tgt[dst_rows] = v[src_index]


def channels(sig):
    return torch.cat([torch.stack([sig['vx'], sig['vy'], sig['r']], -1), sig['omega']], -1)


class System2:
    def __init__(self, env, img, net, mu_probe, period=10, horizon=50, factors=FACTORS, candidates=CANDIDATES, sigma=SIGMA0,
                 forget=0.97, floor=1e-3, q_lo=0.1, bank=True, shield=True):
        """env: the real FastDriftEnv246 (B rows); img: an imagination FastDriftEnv246 on the same courses with at least
        B * max(len(factors), len(candidates)) rows; net: System 1 (ActorCritic: act(obs, deterministic)); mu_probe [B]."""
        self.env, self.img, self.net = env, img, net
        B = env.B; dev = env.dev; self.B = B
        if img.B < B * max(len(factors), len(candidates)):
            raise ValueError('imagination batch too small')
        self.period, self.horizon, self.forget, self.floor, self.q_lo = period, horizon, forget, floor, q_lo
        self.f = torch.as_tensor(factors, dtype=env.dtype, device=dev); self.c = torch.as_tensor(candidates, dtype=env.dtype, device=dev)
        self.sigma = torch.as_tensor(sigma, dtype=env.dtype, device=dev)
        self.mu_probe = mu_probe.to(env.dtype).clone(); self.logp = torch.zeros(B, len(factors), device=dev, dtype=env.dtype)
        self.bank, self.shield = bank, shield
        self.prev = None; self.acts = []; self.obs_sig = []; self.k = 0
        self.log = []

    def posterior(self):
        p = torch.exp(self.logp - self.logp.max(1, keepdim=True).values); return p / p.sum(1, keepdim=True)

    def mu_post(self):
        p = self.posterior(); mu = self.mu_probe[:, None] * self.f[None]
        mean = (p * mu).sum(1); cdf = p.cumsum(1)
        lo = mu.gather(1, torch.searchsorted(cdf, torch.full((self.B, 1), self.q_lo, device=cdf.device, dtype=cdf.dtype)).clamp(max=len(self.f) - 1))[:, 0]
        return mean, lo

    @torch.no_grad()
    def _friction_update(self):
        env, img, B, H = self.env, self.img, self.B, len(self.f)
        rows = torch.arange(H * B, device=env.dev); src = torch.arange(B, device=env.dev).repeat(H)
        load(img, self.prev, rows, src)
        mu_h = (self.mu_probe.repeat(H) * self.f.repeat_interleave(B))[:, None].expand(H * B, 4)
        img.kernel.mu[rows] = mu_h.to(img.dtype); img.mu[rows] = mu_h.to(img.dtype)
        ll = torch.zeros(H * B, device=env.dev, dtype=env.dtype)
        for a, o in zip(self.acts, self.obs_sig):
            act = torch.zeros(img.B, a.shape[1], device=env.dev, dtype=a.dtype); act[:H * B] = a.repeat(H, 1)
            img.step(act, auto_reset=False)
            pred = channels(img.sig)[:H * B]
            ll += -0.5 * (((o.repeat(H, 1) - pred) / self.sigma) ** 2).sum(-1)
        ll = torch.nan_to_num(ll.view(H, B).T, nan=-1e9)
        self.logp = self.forget * self.logp + ll
        self.logp -= self.logp.max(1, keepdim=True).values
        p = torch.clamp(torch.exp(self.logp) / torch.exp(self.logp).sum(1, keepdim=True), min=self.floor)
        self.logp = torch.log(p / p.sum(1, keepdim=True))

    @torch.no_grad()
    def _shield(self, mean, lo):
        env, img, B, C = self.env, self.img, self.B, len(self.c)
        now = snapshot(env); rows = torch.arange(C * B, device=env.dev); src = torch.arange(B, device=env.dev).repeat(C)
        load(img, now, rows, src)
        cand = (mean.repeat(C) * self.c.repeat_interleave(B))
        img.mu_hat[rows] = cand.to(img.dtype)
        world = lo.repeat(C)[:, None].expand(C * B, 4).to(img.dtype)
        img.kernel.mu[rows] = world; img.mu[rows] = world
        crash = torch.zeros(C * B, dtype=torch.bool, device=env.dev); o = img.obs()
        for _ in range(self.horizon):
            a = self.net.act(o, deterministic=True)
            o, p, r, term, trunc, info = img.step(a, auto_reset=False)
            crash |= info['crash'][:C * B]
        ok = ~crash.view(C, B)                                     # candidates descending
        first_ok = torch.where(ok.any(0), ok.float().argmax(0), torch.full((B,), C - 1, device=env.dev))
        return cand.view(C, B).gather(0, first_ok[None])[0], ok

    @torch.no_grad()
    def before_step(self):
        """Called before every real control step; every `period` steps updates friction and sets env.mu_hat."""
        env = self.env
        if self.k % self.period == 0:
            if self.bank and self.prev is not None and self.acts:
                self._friction_update()
            mean, lo = self.mu_post() if self.bank else (self.mu_probe, self.mu_probe)
            if self.shield:
                chosen, ok = self._shield(mean, lo)
            else:
                chosen, ok = mean, None
            env.mu_hat.copy_(chosen.to(env.dtype))
            self.log.append(dict(k=self.k, mean=mean.cpu().numpy(), lo=lo.cpu().numpy(), chosen=chosen.cpu().numpy()))
            self.prev = snapshot(env) if self.bank else None; self.acts = []; self.obs_sig = []
        self.k += 1

    def after_step(self, action):
        if self.bank:
            self.acts.append(action.clone()); self.obs_sig.append(channels(self.env.sig).clone())
