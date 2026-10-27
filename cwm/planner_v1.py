"""Planner."""

from __future__ import annotations

import math

import numpy as np
import torch

from cwm import vehicle_v1 as cs
from cwm.physics_v1 import PlanarPhysics

G = 9.80665


P_REGEN_TOTAL = 4 * 60.0e3      # W: the x8 motor map is constant-power at about 60 kW per motor above 27 rad/s


def envelope_vlim(kap, mu_hat, v_margin, ds, preview_m, mass=1323.9, p_total=P_REGEN_TOTAL, g=G):
    """Allowed speed at every window index so that every corner within preview_m ahead can still be reached at
    v_margin * sqrt(mu_hat g / |kappa|). The braking curve is friction-limited at 0.8 mu_hat g and power-limited by
    motor regeneration (a <= p_total / (m v)); the vehicle has no friction brakes in this action space."""
    n = len(kap)
    vc = v_margin * np.sqrt(mu_hat * g / np.maximum(np.abs(kap), 1e-4))
    a_mu = 0.8 * mu_hat * g
    v_star = p_total / (mass * a_mu)
    steps = np.arange(0, int(preview_m / ds), 10)
    ja = np.clip(np.arange(n)[:, None] + steps[None, :], 0, n - 1)
    d = steps[None, :] * ds
    v0 = vc[ja]
    d1 = np.maximum(v_star ** 2 - v0 ** 2, 0.0) / (2 * a_mu)          # distance of the friction-limited phase
    v_fric = np.sqrt(v0 ** 2 + 2 * a_mu * d)
    v_base = np.maximum(v0, np.minimum(v_star, v_fric))
    d_pow = np.maximum(d - np.where(v0 < v_star, d1, 0.0), 0.0)
    v_pow = np.cbrt(np.maximum(v_base, v0) ** 3 + 3 * p_total * d_pow / mass)
    v_allow = np.where((v0 < v_star) & (d <= d1), v_fric, v_pow)
    return np.min(v_allow, 1)


class MPPI:
    def __init__(self, env_cfg, K=256, T=40, n_sub=5, lam=2.0, sigma=(0.25, 0.25, 0.35), device="cpu",
                 seed=0, preview_m=50.0, w_term=0.05, w_lat_term=0.0, v_margin=1.0, w_spin=0.0,
                 beta_safe_deg=40.0, beta_everywhere=False, world_model="physics2", dtype=torch.float32):
        self.c = env_cfg
        self.K, self.T, self.lam = K, T, lam
        self.dev, self.dt_ = device, dtype
        # world model: planar physics by default; world_model='hybrid:<checkpoint>' uses the learned hybrid
        if world_model.startswith("hybrid:"):
            from cwm.wm_adapters import HybridStepper
            self.ph = HybridStepper(world_model.split(":", 1)[1], device, dtype)
        else:
            self.ph = PlanarPhysics(device, dtype, n_sub=n_sub)
        self.world_model = world_model
        self.layout = env_cfg.layout
        self.na = {"2WS_RWD": 2, "2WS_AWD": 3, "2WS_TV": 5, "4WS_RWD": 3, "4WS_TV": 6}[self.layout]
        sig = [sigma[0]] + ([sigma[1]] if self.layout.startswith("4WS") else []) + \
              [sigma[2]] * (self.na - (2 if self.layout.startswith("4WS") else 1))
        self.sigma = torch.tensor(sig, device=device, dtype=dtype)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.preview_m, self.w_term, self.w_lat_term = preview_m, w_term, w_lat_term
        self.v_margin = v_margin          # fraction of the steady-state corner speed allowed
        self.w_spin, self.beta_safe_deg = w_spin, beta_safe_deg
        self.beta_everywhere = beta_everywhere
        self.U = torch.zeros(T, self.na, device=device, dtype=dtype)
        self.h = None
        self.prev = None

    # ------------------------------------------------------------ helpers
    def _t(self, x):
        return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=self.dt_, device=self.dev)

    def reset(self):
        self.U.zero_()
        self.h = None
        self.prev = None

    def _state(self, env):
        s = env.state
        return self._t(np.concatenate([[s["vx"], s["vy"], s["r"]], s["omega"], s["my_drive"]]))[None]

    def _map(self, a, steer_state, torque_state, omega):
        """Batched DDEVEnv._apply: a (B, na) in [-1, 1] -> commands (B, 8), new steer (B, 2) and torque (B, 4) states."""
        c = self.c
        a = a.clamp(-1.0, 1.0)
        df_t = a[:, 0] * math.radians(c.steer_f_max_deg)
        four = self.layout.startswith("4WS")
        dr_t = a[:, 1] * math.radians(c.steer_r_max_deg) if four else torch.zeros_like(df_t)
        tq = a[:, 2:] if four else a[:, 1:]
        if self.layout.endswith("RWD"):
            T4 = torch.stack([torch.zeros_like(tq[:, 0]), torch.zeros_like(tq[:, 0]), tq[:, 0], tq[:, 0]], 1)
        elif self.layout.endswith("AWD"):
            T4 = torch.stack([tq[:, 0], tq[:, 0], tq[:, 1], tq[:, 1]], 1)
        else:
            T4 = tq
        rf, rr = math.radians(c.steer_rate_f_dps) * c.dt, math.radians(c.steer_rate_r_dps) * c.dt
        sf = steer_state[:, 0] + (df_t - steer_state[:, 0]).clamp(-rf, rf)
        sr = steer_state[:, 1] + (dr_t - steer_state[:, 1]).clamp(-rr, rr)
        rt = c.torque_rate * c.dt
        ts = torque_state + (T4 * c.torque_max - torque_state).clamp(-rt, rt)
        pos = ts > 0
        f = torch.where(pos, ((c.omega_lim - omega) / c.omega_band).clamp(0.0, 1.0),
                        (omega / c.omega_band).clamp(0.0, 1.0))
        cmd = torch.cat([torch.stack([sf, sf, sr, sr], 1), ts * f], 1)
        return cmd, torch.stack([sf, sr], 1), ts

    def observe(self, env):
        """Advance the physics hidden state over the step just taken: from the previously observed state with the
        command that was applied during that step (env.cmd now), then re-anchor on the observed state."""
        s = self._state(env)
        mu = self._t([env.mu_hat if env.mu_hat is not None else 0.65])
        u_applied = self._t(np.concatenate([env.cmd.steer, env.cmd.motor]))[None]
        if self.h is None:
            self.h = self.ph.init_hidden(s, u_applied, mu, self._t([env.state["ax"]]), self._t([env.state["ay"]]))
        else:
            _, self.h = self.ph.step(self.prev, self.h, u_applied, mu)
        self.prev = s

    # ------------------------------------------------------------ planning
    @torch.no_grad()
    def act(self, env) -> np.ndarray:
        c, K, T = self.c, self.K, self.T
        self.observe(env)
        mu_hat = float(env.mu_hat if env.mu_hat is not None else 0.65)
        eps = torch.randn(K, T, self.na, generator=self.gen, device=self.dev, dtype=self.dt_) * self.sigma
        eps[0] = 0.0                                             # keep the nominal sequence
        A = (self.U[None] + eps).clamp(-1.0, 1.0)
        G_, alive = self._rollout(env, A, mu_hat)
        w = torch.softmax((G_ - G_.max()) / self.lam, 0)
        self.U = (w[:, None, None] * A).sum(0)
        a0 = self.U[0].cpu().numpy().astype(np.float64)
        self.U = torch.cat([self.U[1:], self.U[-1:]], 0)
        self.last = dict(G_best=float(G_.max()), G_mean=float(G_.mean()), alive=float(alive.float().mean()),
                         ess=float(1.0 / (w ** 2).sum()))
        return a0

    @torch.no_grad()
    def _rollout(self, env, A, mu_hat):
        """Returns (G, alive) for action sequences A (K, T, na) from the current observed state."""
        c, K, T = self.c, self.K, self.T
        tr = env.track
        mu = torch.full((K,), mu_hat, device=self.dev, dtype=self.dt_)
        # initial conditions, replicated
        s = self.prev.expand(K, -1).clone()
        h = self.h.expand(K, -1).clone()
        st = self._t(env.steer_state)[None].expand(K, -1).clone()
        ts = self._t(env.torque_state)[None].expand(K, -1).clone()
        x = torch.full((K,), float(env.state["x"]), device=self.dev, dtype=self.dt_)
        y = torch.full((K,), float(env.state["y"]), device=self.dev, dtype=self.dt_)
        yaw = torch.full((K,), float(env.state["yaw"]), device=self.dev, dtype=self.dt_)
        a_prev = self._t(env.prev_a)[None].expand(K, -1)
        # local track window
        n = len(tr.s)
        span = int(min(n - 1 if tr.closed else n, (40.0 * T * c.dt + self.preview_m + 20.0) / tr.ds))
        idx = (np.arange(env.idx - 10, env.idx - 10 + span) % n) if tr.closed else \
            np.clip(np.arange(env.idx - 10, env.idx - 10 + span), 0, n - 1)
        txy = self._t(tr.xy[idx])
        tpsi = self._t(tr.psi[idx])
        ts_ = self._t(tr.s[idx])
        if tr.closed:
            ts_ = self._t(np.unwrap(tr.s[idx] * 2 * np.pi / tr.length) * tr.length / (2 * np.pi))
        kap = tr.kappa[idx]
        z = np.clip((np.abs(kap) - c.drift_kappa_min) / c.drift_kappa_min, 0.0, 1.0)
        if c.task == "drift":
            bref = self._t(-np.sign(kap) * np.radians(c.beta_ref_deg) * z)
        elif self.beta_everywhere:            # race: small sideslip everywhere (beta_ref 0)
            bref = self._t(np.zeros_like(kap))
        else:
            bref = None
        s_prev = torch.full((K,), float(ts_[10]), device=self.dev, dtype=self.dt_)
        # project the current position to set s_prev properly
        d0 = (txy[:, 0] - x[0]) ** 2 + (txy[:, 1] - y[0]) ** 2
        j0 = int(torch.argmin(d0))
        s_prev[:] = ts_[j0]
        G_ = torch.zeros(K, device=self.dev, dtype=self.dt_)
        alive = torch.ones(K, device=self.dev, dtype=torch.bool)
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
            ds = s_tr - s_prev
            s_prev = s_tr
            p = ds / (c.dt * 20.0)
            q = torch.exp(-(e_lat / 2.0) ** 2)
            if bref is not None:
                br = bref[j]
                zone = br.abs() > 1e-6
                if self.beta_everywhere:          # straights and gentle bends: keep the car straight (beta_ref 0)
                    zone = torch.ones_like(zone)
                q = q * torch.where(zone, torch.exp(-((beta - br) / math.radians(15.0)) ** 2), torch.ones_like(q))
            da = A[:, t] - (a_prev if t == 0 else A[:, t - 1])
            r = p * q - 0.01 * (da ** 2).sum(1)
            if self.w_spin > 0:           # barrier well before the 70 deg spin termination
                r = r - self.w_spin * ((beta.abs() - math.radians(self.beta_safe_deg)).clamp(min=0.0)) ** 2
            dead = (e_lat.abs() > c.width + c.offtrack_margin) | (beta.abs() > math.radians(70.0)) | \
                   (e_psi.abs() > math.radians(110.0)) | ~torch.isfinite(s).all(1)
            r = torch.where(dead & alive, r - 10.0, r)
            G_ = G_ + torch.where(alive, r, torch.zeros_like(r))
            alive = alive & ~dead
        # friction-envelope terminal cost on the survivors
        if self.w_term > 0:
            vlim = self._t(envelope_vlim(kap, mu_hat, self.v_margin, tr.ds, self.preview_m))[j]
            speed = torch.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
            G_ = G_ - self.w_term * torch.where(alive, (speed - vlim).clamp(min=0.0) ** 2, torch.zeros_like(G_))
        if self.w_lat_term > 0:            # end the horizon near the centerline, velocity along the track
            # course angle relative to the track = e_psi + beta (a drift keeps it near zero while beta is large)
            G_ = G_ - self.w_lat_term * torch.where(alive, e_lat ** 2 + (5.0 * torch.sin(e_psi + beta)) ** 2,
                                                    torch.zeros_like(G_))
        G_ = torch.nan_to_num(G_, nan=-1e6, neginf=-1e6, posinf=-1e6)
        return G_, alive
