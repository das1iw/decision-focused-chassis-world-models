"""CUDA-graph MPPI: the planner of mppi.MPPI with the whole sample rollout captured as one CUDA graph.

Same objective, model and action mapping as mppi.MPPI (the CPU reference). Per control step, the host code:
- fills static input tensors (state, physics hidden, actuator states, pose, nominal plan, noise, track window);
- replays the graph;
- reads back the first action.
The friction-envelope terminal speed limit is precomputed per window index in torch, so the graph contains no
host synchronization.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from trfc_drift.mppi import MPPI, envelope_vlim

G = 9.80665


class MPPIGraph(MPPI):
    def __init__(self, env_cfg, K=512, T=50, n_sub=5, span=260, **kw):
        kw.setdefault("device", "cuda")
        super().__init__(env_cfg, K=K, T=T, n_sub=n_sub, **kw)
        dev, dt = self.dev, self.dt_
        z = lambda *shape: torch.zeros(*shape, device=dev, dtype=dt)  # noqa: E731
        self.span = span
        self.i_s, self.i_h, self.i_st, self.i_ts = z(1, 11), z(1, 18), z(1, 2), z(1, 4)
        self.i_pose, self.i_aprev, self.i_eps = z(3), z(1, self.na), z(K, T, self.na)
        self.i_U, self.i_mu = z(T, self.na), z(1)
        self.i_txy, self.i_tpsi, self.i_ts_, self.i_bref, self.i_vl = z(span, 2), z(span), z(span), z(span), z(span)
        self.i_s0 = z(1)
        self.graph = None

    def _rollout(self):
        c, K, T = self.c, self.K, self.T
        A = (self.i_U[None] + self.i_eps).clamp(-1.0, 1.0)
        s = self.i_s.expand(K, -1).clone()
        h = self.i_h.expand(K, -1).clone()
        st = self.i_st.expand(K, -1).clone()
        ts = self.i_ts.expand(K, -1).clone()
        x = self.i_pose[0].expand(K).clone()
        y = self.i_pose[1].expand(K).clone()
        yaw = self.i_pose[2].expand(K).clone()
        mu = self.i_mu.expand(K)
        txy, tpsi, ts_, bref = self.i_txy, self.i_tpsi, self.i_ts_, self.i_bref
        s_prev = self.i_s0.expand(K).clone()
        G_ = torch.zeros(K, device=self.dev, dtype=self.dt_)
        alive = torch.ones(K, device=self.dev, dtype=torch.bool)
        drift = c.task == "drift"
        for t in range(T):
            cmd, st, ts = self._map(A[:, t], st, ts, s[:, 3:7])
            s_n, h = self.ph.step(s, h, cmd, mu)
            vxm, vym, rm = 0.5 * (s[:, 0] + s_n[:, 0]), 0.5 * (s[:, 1] + s_n[:, 1]), 0.5 * (s[:, 2] + s_n[:, 2])
            yaw_m = yaw + 0.5 * c.dt * rm
            x = x + c.dt * (vxm * torch.cos(yaw_m) - vym * torch.sin(yaw_m))
            y = y + c.dt * (vxm * torch.sin(yaw_m) + vym * torch.cos(yaw_m))
            yaw = yaw + c.dt * rm
            s = s_n
            d2 = (x[:, None] - txy[None, :, 0]) ** 2 + (y[:, None] - txy[None, :, 1]) ** 2
            j = torch.argmin(d2, 1)
            tx, ty = torch.cos(tpsi[j]), torch.sin(tpsi[j])
            dx, dy = x - txy[j, 0], y - txy[j, 1]
            e_lat = -dx * ty + dy * tx
            s_tr = ts_[j] + dx * tx + dy * ty
            e_psi = torch.atan2(torch.sin(yaw - tpsi[j]), torch.cos(yaw - tpsi[j]))
            beta = torch.atan2(s[:, 1], s[:, 0].clamp(min=0.5))
            p = (s_tr - s_prev) / (c.dt * 20.0)
            s_prev = s_tr
            q = torch.exp(-(e_lat / 2.0) ** 2)
            if drift:
                br = bref[j]
                zone = torch.ones_like(br, dtype=torch.bool) if self.beta_everywhere else br.abs() > 1e-6
                q = q * torch.where(zone, torch.exp(-((beta - br) / math.radians(15.0)) ** 2), torch.ones_like(q))
            da = A[:, t] - (self.i_aprev.expand(K, -1) if t == 0 else A[:, t - 1])
            r = p * q - 0.01 * (da ** 2).sum(1)
            if self.w_spin > 0:           # barrier well before the 70 deg spin termination
                r = r - self.w_spin * ((beta.abs() - math.radians(self.beta_safe_deg)).clamp(min=0.0)) ** 2
            dead = (e_lat.abs() > c.width + c.offtrack_margin) | (beta.abs() > math.radians(70.0)) | \
                   (e_psi.abs() > math.radians(110.0)) | ~torch.isfinite(s).all(1)
            r = torch.where(dead & alive, r - 10.0, r)
            G_ = G_ + torch.where(alive, r, torch.zeros_like(r))
            alive = alive & ~dead
        if self.w_term > 0:
            speed = torch.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
            G_ = G_ - self.w_term * torch.where(alive, (speed - self.i_vl[j]).clamp(min=0.0) ** 2, torch.zeros_like(G_))
        if self.w_lat_term > 0:
            G_ = G_ - self.w_lat_term * torch.where(alive, e_lat ** 2 + (5.0 * torch.sin(e_psi + beta)) ** 2,
                                                    torch.zeros_like(G_))
        G_ = torch.nan_to_num(G_, nan=-1e6, neginf=-1e6, posinf=-1e6)
        w = torch.softmax((G_ - G_.max()) / self.lam, 0)
        self.o_U = (w[:, None, None] * A).sum(0)
        self.o_stats = torch.stack([G_.max(), G_.mean(), alive.float().mean(), 1.0 / (w ** 2).sum()])

    def _fill(self, env):
        c, tr = self.c, env.track
        self.i_s.copy_(self.prev)
        self.i_h.copy_(self.h)
        self.i_st.copy_(self._t(env.steer_state)[None])
        self.i_ts.copy_(self._t(env.torque_state)[None])
        self.i_pose.copy_(self._t([env.state["x"], env.state["y"], env.state["yaw"]]))
        self.i_aprev.copy_(self._t(env.prev_a)[None])
        self.i_U.copy_(self.U)
        self.i_eps.copy_(torch.randn(self.K, self.T, self.na, generator=self.gen, device=self.dev, dtype=self.dt_)
                         * self.sigma)
        self.i_eps[0].zero_()
        mu_hat = float(env.mu_hat if env.mu_hat is not None else 0.65)
        self.i_mu.fill_(mu_hat)
        n = len(tr.s)
        rng = np.arange(env.idx - 10, env.idx - 10 + self.span)
        idx = rng % n if tr.closed else np.clip(rng, 0, n - 1)
        self.i_txy.copy_(self._t(tr.xy[idx]))
        self.i_tpsi.copy_(self._t(tr.psi[idx]))
        sw = tr.s[idx].astype(float)
        if tr.closed:
            sw = np.unwrap(sw * 2 * np.pi / tr.length) * tr.length / (2 * np.pi)
        self.i_ts_.copy_(self._t(sw))
        kap = tr.kappa[idx]
        zz = np.clip((np.abs(kap) - c.drift_kappa_min) / c.drift_kappa_min, 0.0, 1.0)
        self.i_bref.copy_(self._t(-np.sign(kap) * np.radians(c.beta_ref_deg) * zz))
        # envelope speed limit ahead of every window index
        vl = envelope_vlim(kap, mu_hat, self.v_margin, tr.ds, self.preview_m)
        self.i_vl.copy_(self._t(vl))
        d0 = (tr.xy[idx, 0] - env.state["x"]) ** 2 + (tr.xy[idx, 1] - env.state["y"]) ** 2
        self.i_s0.fill_(float(sw[int(np.argmin(d0))]))

    @torch.no_grad()
    def act(self, env) -> np.ndarray:
        self.observe(env)
        self._fill(env)
        if self.graph is None:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                self._rollout()
            torch.cuda.current_stream().wait_stream(st)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._rollout()
        self.graph.replay()
        U = self.o_U.clone()
        a0 = U[0].cpu().numpy().astype(np.float64)
        self.U = torch.cat([U[1:], U[-1:]], 0)
        st = self.o_stats.cpu().numpy()
        self.last = dict(G_best=float(st[0]), G_mean=float(st[1]), alive=float(st[2]), ess=float(st[3]))
        return a0
