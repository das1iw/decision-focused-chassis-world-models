"""MPPI + SAC: world-model planning with a learned long-horizon value and a learned proposal (research/105).

Plain MPPI (mppi.MPPI) looks about 1 s ahead with the physics-v2 world model and ends the horizon with hand-made
terminal costs. Here a trained SAC agent of the same layout and information set supplies two things:
- terminal value: G += gamma^T w_v min(Q1, Q2)(o_T, tanh(mu_pi(o_T))) for samples still alive, so the planner values
  what happens after its horizon the way the RL agent learned to;
- proposal: the first K_pi samples take the SAC policy's stochastic action at every rollout step instead of the
  nominal plan plus noise, so good policy behaviour is always among the candidates.
Observations o_t for predicted states are built in torch exactly like DDEVEnv._obs (same order and scaling), from the
world-model state, the sampled actions, the track window and the same envelope LoadModel (frozen calibration):
- body-frame ax, ay from the predicted velocity change;
- realized steer approximated by the commanded road-wheel angle (compliance is below 0.3 deg).

The SAC checkpoint is read-only (final.pt of a training run). Hand-made terminal costs of MPPI stay available and can
be switched off (w_term = w_lat_term = 0) so the value function does that job.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from trfc_drift import friction as F
from trfc_drift.envelope import LoadModel
from trfc_drift.mppi import MPPI, envelope_vlim
from trfc_drift.sac import Actor, Critic

G = 9.80665


class SACParts:
    def __init__(self, ckpt_path, n_obs, n_act, device="cpu", dtype=torch.float32):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        self.actor = Actor(n_obs, n_act, cfg["hidden"], cfg["layers"])
        self.actor.load_state_dict(ck["actor"])
        self.critic = Critic(n_obs, n_act, cfg["hidden"], cfg["layers"])
        self.critic.load_state_dict(ck["critic"])
        for m in (self.actor, self.critic):
            m.to(device=device, dtype=dtype).eval()
            for p in m.parameters():
                p.requires_grad_(False)
        self.gamma = float(cfg["gamma"])


class ValueMPPI(MPPI):
    def __init__(self, env_cfg, sac_ckpt, K=256, T=50, n_sub=5, K_pi=64, w_v=1.0, pi_noise=1.0, **kw):
        super().__init__(env_cfg, K=K, T=T, n_sub=n_sub, **kw)
        from trfc_drift.env import obs_dim
        self.nobs = obs_dim(env_cfg)
        self.sac = SACParts(sac_ckpt, self.nobs, self.na, self.dev, self.dt_)
        self.K_pi, self.w_v, self.pi_noise = K_pi, w_v, pi_noise
        lm = LoadModel.load()
        self.lm_static = self._t(lm.fz_static)
        self.lm_h, self.lm_phi = lm.h, lm.phi_f
        self.pv = np.array(env_cfg.preview_m)

    # ---------------------------------------------------------------- observation builder (DDEVEnv._obs in torch)
    def _env_fz(self, ax, ay):
        from trfc_drift import carsim_dll as cs
        m, L, h = F.M, F.L, self.lm_h
        dlong = m * ax * h / L / 2.0
        dlf = m * ay * h * self.lm_phi / cs.VEH["track_f"]
        dlr = m * ay * h * (1.0 - self.lm_phi) / cs.VEH["track_r"]
        return self.lm_static + torch.stack([-dlong - dlf / 2, -dlong + dlf / 2, dlong - dlr / 2, dlong + dlr / 2], 1)

    def _obs(self, s, s_prev, a_prev, steer_cmd, e_lat, e_psi, jw, kap_w, bref_w, mu_hat):
        c = self.c
        dt = c.dt
        vx, vy, r = s[:, 0], s[:, 1], s[:, 2]
        ax = (s[:, 0] - s_prev[:, 0]) / dt - vy * r
        ay = (s[:, 1] - s_prev[:, 1]) / dt + vx * r
        beta = torch.atan2(vy, vx.clamp(min=0.5))
        om = s[:, 3:7]
        ratio = (om * F.R / vx.clamp(min=1.0)[:, None] - 1.0).clamp(-1.0, 2.0)
        st = torch.stack([0.5 * (steer_cmd[:, 0] + steer_cmd[:, 1]), 0.5 * (steer_cmd[:, 2] + steer_cmd[:, 3])], 1) * 3.0
        n = kap_w.shape[0]
        offs = torch.as_tensor(np.round(self.pv / 0.5).astype(np.int64), device=self.dev)
        idx = (jw[:, None] + offs[None, :]).clamp(max=n - 1)
        parts = [torch.stack([vx / 20.0, vy / 5.0, r, ax / 10.0, ay / 10.0, beta], 1), ratio, st, a_prev,
                 torch.stack([e_lat / c.width, torch.sin(e_psi), torch.cos(e_psi)], 1), (kap_w[idx] * 20.0).clamp(-2, 2)]
        if c.task == "drift":
            parts.append(bref_w[idx] * 2.0)
        if c.friction_info != "none":
            parts.append(torch.full_like(vx, mu_hat)[:, None])
            if c.friction_info.endswith("_env"):
                fz = self._env_fz(ax, ay).clamp(min=0.0)
                cap = mu_hat * fz
                omdot = (om - s_prev[:, 3:7]) / dt
                fx = (s[:, 7:11] - F.JW * omdot) / F.R
                lat = torch.sqrt((cap ** 2 - fx ** 2).clamp(min=0.0))
                parts.append(torch.cat([cap, fx, lat], 1) / (F.M * G / 4.0))
        return torch.nan_to_num(torch.cat(parts, 1), nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------------- planning
    @torch.no_grad()
    def act(self, env) -> np.ndarray:
        c, K, T = self.c, self.K, self.T
        self.observe(env)
        tr = env.track
        mu_hat = float(env.mu_hat if env.mu_hat is not None else 0.65)
        mu = torch.full((K,), mu_hat, device=self.dev, dtype=self.dt_)
        eps = torch.randn(K, T, self.na, generator=self.gen, device=self.dev, dtype=self.dt_) * self.sigma
        eps[self.K_pi] = 0.0                                     # the nominal sequence (first non-policy sample)
        A = (self.U[None] + eps).clamp(-1.0, 1.0)
        s = self.prev.expand(K, -1).clone()
        h = self.h.expand(K, -1).clone()
        st = self._t(env.steer_state)[None].expand(K, -1).clone()
        ts = self._t(env.torque_state)[None].expand(K, -1).clone()
        x = torch.full((K,), float(env.state["x"]), device=self.dev, dtype=self.dt_)
        y = torch.full((K,), float(env.state["y"]), device=self.dev, dtype=self.dt_)
        yaw = torch.full((K,), float(env.state["yaw"]), device=self.dev, dtype=self.dt_)
        a_prev = self._t(env.prev_a)[None].expand(K, -1).clone()
        n = len(tr.s)
        span = int(min(n - 1 if tr.closed else n, (40.0 * T * c.dt + max(self.preview_m, 80.0) + 20.0) / tr.ds))
        rng_i = np.arange(env.idx - 10, env.idx - 10 + span)
        idx = rng_i % n if tr.closed else np.clip(rng_i, 0, n - 1)
        txy, tpsi = self._t(tr.xy[idx]), self._t(tr.psi[idx])
        sw = tr.s[idx].astype(float)
        if tr.closed:
            sw = np.unwrap(sw * 2 * np.pi / tr.length) * tr.length / (2 * np.pi)
        ts_ = self._t(sw)
        kap = tr.kappa[idx]
        kap_w = self._t(kap)
        zz = np.clip((np.abs(kap) - c.drift_kappa_min) / c.drift_kappa_min, 0.0, 1.0)
        bref = self._t(-np.sign(kap) * np.radians(c.beta_ref_deg) * zz)
        d0 = (tr.xy[idx, 0] - env.state["x"]) ** 2 + (tr.xy[idx, 1] - env.state["y"]) ** 2
        s_prev_tr = torch.full((K,), float(sw[int(np.argmin(d0))]), device=self.dev, dtype=self.dt_)
        G_ = torch.zeros(K, device=self.dev, dtype=self.dt_)
        alive = torch.ones(K, device=self.dev, dtype=torch.bool)
        # observation at the current state for the policy's first action (real observation from the env)
        o_now = self._t(env._obs())[None].expand(K, -1)
        o_t = o_now
        e_lat = e_psi = beta = None
        j = None
        for t in range(T):
            if self.K_pi > 0:
                mu_a, log_std = self.sac.actor(o_t[:self.K_pi])
                a_pi = torch.tanh(mu_a + self.pi_noise * log_std.exp() *
                                  torch.randn(mu_a.shape, generator=self.gen, device=self.dev, dtype=self.dt_))
                A[:self.K_pi, t] = a_pi
            cmd, st, ts = self._map(A[:, t], st, ts, s[:, 3:7])
            s_n, h = self.ph.step(s, h, cmd, mu)
            vxm, vym, rm = 0.5 * (s[:, 0] + s_n[:, 0]), 0.5 * (s[:, 1] + s_n[:, 1]), 0.5 * (s[:, 2] + s_n[:, 2])
            yaw_m = yaw + 0.5 * c.dt * rm
            x = x + c.dt * (vxm * torch.cos(yaw_m) - vym * torch.sin(yaw_m))
            y = y + c.dt * (vxm * torch.sin(yaw_m) + vym * torch.cos(yaw_m))
            yaw = yaw + c.dt * rm
            s_old, s = s, s_n
            d2 = (x[:, None] - txy[None, :, 0]) ** 2 + (y[:, None] - txy[None, :, 1]) ** 2
            j = torch.argmin(d2, 1)
            tx, ty = torch.cos(tpsi[j]), torch.sin(tpsi[j])
            dx, dy = x - txy[j, 0], y - txy[j, 1]
            e_lat = -dx * ty + dy * tx
            s_tr = ts_[j] + dx * tx + dy * ty
            e_psi = torch.atan2(torch.sin(yaw - tpsi[j]), torch.cos(yaw - tpsi[j]))
            beta = torch.atan2(s[:, 1], s[:, 0].clamp(min=0.5))
            p = (s_tr - s_prev_tr) / (c.dt * 20.0)
            s_prev_tr = s_tr
            q = torch.exp(-(e_lat / 2.0) ** 2)
            if c.task == "drift":
                br = bref[j]
                zone = torch.ones_like(br, dtype=torch.bool) if self.beta_everywhere else br.abs() > 1e-6
                q = q * torch.where(zone, torch.exp(-((beta - br) / math.radians(15.0)) ** 2), torch.ones_like(q))
            da = A[:, t] - a_prev
            r = p * q - 0.01 * (da ** 2).sum(1)
            if self.w_spin > 0:
                r = r - self.w_spin * ((beta.abs() - math.radians(self.beta_safe_deg)).clamp(min=0.0)) ** 2
            dead = (e_lat.abs() > c.width + c.offtrack_margin) | (beta.abs() > math.radians(70.0)) | \
                   (e_psi.abs() > math.radians(110.0)) | ~torch.isfinite(s).all(1)
            r = torch.where(dead & alive, r - 10.0, r)
            G_ = G_ + torch.where(alive, r, torch.zeros_like(r)) * (self.sac.gamma ** t)
            alive = alive & ~dead
            a_prev = A[:, t]
            o_t = self._obs(s, s_old, a_prev, cmd[:, 0:4], e_lat, e_psi, j, kap_w, bref, mu_hat)
        # learned terminal value (SAC critic at the policy's mean action)
        if self.w_v > 0:
            mu_T, _ = self.sac.actor(o_t)
            q1, q2 = self.sac.critic(o_t, torch.tanh(mu_T))
            V = torch.minimum(q1, q2)
            G_ = G_ + torch.where(alive, self.w_v * (self.sac.gamma ** T) * V, torch.zeros_like(V))
        if self.w_term > 0:
            speed = torch.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
            vlim = self._t(envelope_vlim(kap, mu_hat, self.v_margin, tr.ds, self.preview_m))[j]
            G_ = G_ - self.w_term * torch.where(alive, (speed - vlim).clamp(min=0.0) ** 2, torch.zeros_like(G_))
        if self.w_lat_term > 0:
            G_ = G_ - self.w_lat_term * torch.where(alive, e_lat ** 2 + (5.0 * torch.sin(e_psi + beta)) ** 2,
                                                    torch.zeros_like(G_))
        G_ = torch.nan_to_num(G_, nan=-1e6, neginf=-1e6, posinf=-1e6)
        w = torch.softmax((G_ - G_.max()) / self.lam, 0)
        U = (w[:, None, None] * A).sum(0)
        a0 = U[0].cpu().numpy().astype(np.float64)
        self.U = torch.cat([U[1:], U[-1:]], 0)
        wp = float(w[:self.K_pi].sum()) if self.K_pi > 0 else 0.0
        self.last = dict(G_best=float(G_.max()), G_mean=float(G_.mean()), alive=float(alive.float().mean()),
                         ess=float(1.0 / (w ** 2).sum()), w_policy=wp)
        return a0
