"""Friction filter."""

from __future__ import annotations

import numpy as np
import torch

from cwm.planner_cpu_v1 import _step

FACTORS = np.round(np.arange(0.70, 1.1001, 0.05), 3)
# one-window (0.2 s) prediction noise per channel: vx, vy, r, omega x4 (m/s, m/s, rad/s, rad/s): planar-physics replay
SIGMA0 = np.array([0.0094, 0.0363, 0.0145, 1.2352, 0.8470, 2.3800, 1.7767])


class MuFilter:
    def __init__(self, gym, mu_probe: float, factors=FACTORS, sigma=SIGMA0, forget: float = 0.97,
                 floor: float = 1e-3, corrector=None, members=(0,)):
        """gym: a WMGym-like object providing .MP (packed physics), .EP (n_sub at [14]), .cfg.dt."""
        self.gym = gym
        self.mu_probe = float(mu_probe)
        self.f = np.asarray(factors, dtype=np.float64)
        self.logp = np.zeros(len(self.f))              # log posterior (flat prior)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.forget, self.floor = forget, floor
        self.corrector, self.members = corrector, list(members)
        self.win = None                                # (s0, h0, [u_1..u_K])
        self.hist = []

    @staticmethod
    def obs_channels(s):
        return np.array([s[0], s[1], s[2], s[3], s[4], s[5], s[6]])

    def start_window(self, s0, h0):
        self.win = (np.asarray(s0, float).copy(), np.asarray(h0, float).copy(), [])

    def add_command(self, u):
        if self.win is not None:
            self.win[2].append(np.asarray(u, float).copy())

    def _replay(self, s0, h0, U, mu):
        n_sub, dt = int(self.gym.EP[14]), self.gym.cfg.dt
        s, h = s0.copy(), h0.copy()
        for u in U:
            s_n, h_n = _step(s.copy(), h.copy(), u.copy(), mu, n_sub, dt, *self.gym.MP)
            if self.corrector is not None:
                corr = np.mean([self.corrector(s[None], h[None], u[None], np.array([mu]), np.array([m]))[0]
                                for m in self.members], 0)
                s_n = s_n + corr
            s, h = s_n, h_n
        return s

    def update(self, s_obs):
        """Close the current window with the observed state at its end; returns the posterior summary."""
        if self.win is None or not self.win[2]:
            return self.summary()
        s0, h0, U = self.win
        o = self.obs_channels(s_obs)
        ll = np.empty(len(self.f))
        for j, f in enumerate(self.f):
            p = self.obs_channels(self._replay(s0, h0, U, self.mu_probe * f))
            ll[j] = -0.5 * float(np.sum(((o - p) / self.sigma) ** 2))
        self.logp = self.forget * self.logp + ll
        self.logp -= self.logp.max()
        post = np.exp(self.logp)
        post = np.maximum(post / post.sum(), self.floor)
        post /= post.sum()
        self.logp = np.log(post)
        self.win = None
        out = self.summary()
        self.hist.append(dict(out, ll_spread=float(ll.max() - ll.min())))
        return out

    def posterior(self):
        p = np.exp(self.logp - self.logp.max())
        return p / p.sum()

    def summary(self, q: float = 0.1):
        p = self.posterior()
        mu = self.mu_probe * self.f
        cdf = np.cumsum(p)
        return dict(mean=float(np.dot(p, mu)), q_lo=float(mu[int(np.searchsorted(cdf, q))]),
                    map=float(mu[int(np.argmax(p))]))


def calibrate_sigma(ds_path, n_windows=3000, K=10, seed=0):
    """Per-channel RMS of the planar-physics K-step replay error on the wm2 dataset (recorded commands), for SIGMA0."""
    from cwm.planner_cpu_v1 import pack_model
    from cwm.physics_v1 import PlanarPhysics
    z = np.load(ds_path)
    S, Hh, U, MU, ep, k = (z[n] for n in ("s", "h", "u", "mu", "ep", "k"))   # decompress once
    ph = PlanarPhysics("cpu", torch.float64, n_sub=10)
    MP = pack_model(ph)
    rng = np.random.default_rng(seed)
    errs = []
    n = len(ep)
    for _ in range(n_windows):
        i = int(rng.integers(0, n - K - 1))
        if ep[i + K] != ep[i] or k[i + K] - k[i] != K:
            continue
        s, h = S[i].copy(), Hh[i].copy()
        for t in range(K):
            s, h = _step(s.copy(), h.copy(), U[i + t].copy(), float(MU[i + t]), 10, 0.02, *MP)
        s_true = S[i + K]
        errs.append(MuFilter.obs_channels(s) - MuFilter.obs_channels(s_true))
    e = np.array(errs)
    return np.sqrt((e ** 2).mean(0)), len(e)
