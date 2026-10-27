"""The 241 environment with the T-IV planar world model physics v2 as the plant (research/241, comparison arm).

Identical task, track pool, actuator layer (power budget), observation layout, rewards and terminations to
trfc_drift/chassis_env241_v1.ChassisEnv241. Only the plant and the way signals are obtained differ:
- plant: trfc_drift/wm_physics2.DDEVPhysics2 (frozen T-IV model: planar body, Bakker tire with lagged slips,
  identified loads, compliance steer and aero), 40 substeps per 20 ms, captured as one CUDA graph per control step;
- pose: the CG is integrated with the mean body velocity over the step; the vehicle origin lies A_CG ahead of the CG;
- Ax/Ay: the mean yaw-frame acceleration over the 20 ms step (physics v2 does not expose instantaneous forces);
- Steer: physics v2's composed angle (command plus identified compliance); omega: physics v2's wheel spin.
Privileged extras use physics v2's lagged slips (tan alpha, kappa) and loads. Friction is the per-row mean of the
wheel values (physics v2 takes one mu per sample).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from trfc_drift.chassis_env241_v1 import ChassisEnv241, TrackPool, EnvConfig241, NACT
from trfc_drift import wm_physics2 as P2


class _Buffers:
    """Command buffers with the kernel interface used by ChassisEnv241.step (replay, bad, y)."""

    def __init__(self, env, B, dt_, dev):
        z = lambda *s: torch.zeros(*s, dtype=dt_, device=dev)
        self.env = env
        self.steer = z(B, 4); self.motor = z(B, 4); self.brake = z(B, 4); self.external = z(B, 4); self.mu = z(B, 4) + 1.
        self.bad = torch.zeros(B, dtype=torch.uint8, device=dev)

    @property
    def y(self):
        return self.env.s

    def replay(self):
        self.env._advance()


class Physics2Env241(ChassisEnv241):
    def __init__(self, tracks, batch, cfg: EnvConfig241 | None = None, *, seed=0, device='cuda', graph=True):
        self.cfg = cfg = cfg or EnvConfig241(); self.B = B = int(batch); self.dev = torch.device(device)
        self.dtype = dt_ = cfg.dtype
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(int(seed))
        self.ph = P2.DDEVPhysics2(str(self.dev), dt_, n_sub=cfg.n_sub)
        self.pool = TrackPool(tracks, self.dev, dt_)
        self.kernel = _Buffers(self, B, dt_, self.dev)
        z = lambda *s: torch.zeros(*s, dtype=dt_, device=self.dev)
        self.s = z(B, 11); self.h = z(B, P2.DDEVPhysics2.NH); self.pose = z(B, 3)          # CG x, y, yaw
        self.acc = z(B, 2)
        self.tid = torch.zeros(B, dtype=torch.long, device=self.dev); self.idx = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.s_tr = z(B); self.s_prog = z(B); self.t = z(B); self.slow_t = z(B); self.laps = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.prev_a = z(B, NACT); self.steer_state = z(B, 2); self.torque_state = z(B, 4); self.mu = z(B, 4); self.mu_hat = z(B)
        self.e_lat = z(B); self.e_psi = z(B); self.sig = {}
        self.preview = torch.as_tensor(cfg.preview_m, dtype=dt_, device=self.dev)
        self.graph = None
        if graph and self.dev.type == 'cuda':
            self._capture()

    # ---------------------------------------------------------------- plant step
    def _body(self):
        k = self.kernel; s0 = self.s; mu = k.mu.mean(1)
        u = torch.cat([k.steer, k.motor], 1)
        s1, h1 = self.ph.step(s0, self.h, u, mu, self.cfg.dt)
        dt = self.cfg.dt; yaw0 = self.pose[:, 2]
        rm = .5 * (s0[:, 2] + s1[:, 2]); yaw1 = yaw0 + rm * dt; ym = yaw0 + .5 * rm * dt
        vx, vy = .5 * (s0[:, 0] + s1[:, 0]), .5 * (s0[:, 1] + s1[:, 1])
        c, sn = torch.cos(ym), torch.sin(ym)
        x1 = self.pose[:, 0] + (c * vx - sn * vy) * dt; y1 = self.pose[:, 1] + (sn * vx + c * vy) * dt
        ax = (s1[:, 0] - s0[:, 0]) / dt - rm * vy; ay = (s1[:, 1] - s0[:, 1]) / dt + rm * vx
        bad = ~(torch.isfinite(s1).all(1) & torch.isfinite(h1).all(1))
        self.s.copy_(s1); self.h.copy_(h1); self.pose.copy_(torch.stack([x1, y1, yaw1], 1)); self.acc.copy_(torch.stack([ax, ay], 1))
        k.bad.copy_(k.bad | bad.to(torch.uint8))

    def _capture(self):
        snap = [x.clone() for x in (self.s, self.h, self.pose, self.acc, self.kernel.bad)]
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(st)
        for x, v in zip((self.s, self.h, self.pose, self.acc, self.kernel.bad), snap):
            x.copy_(v)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self._body()
        for x, v in zip((self.s, self.h, self.pose, self.acc, self.kernel.bad), snap):
            x.copy_(v)

    def _advance(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            with torch.no_grad():
                self._body()

    # ---------------------------------------------------------------- signals and resets
    def signals(self):
        s, h = self.s, self.h; ph = self.ph
        fz = ph.fz(h[:, 8], h[:, 9]); steer = ph.steer(self.kernel.steer, h[:, 10:14], h[:, 14:18], fz, h[:, 9])
        yaw = self.pose[:, 2]
        x = self.pose[:, 0] + P2.A_CG * torch.cos(yaw); y = self.pose[:, 1] + P2.A_CG * torch.sin(yaw)
        return dict(vx=s[:, 0], vy=s[:, 1], r=s[:, 2], ax=self.acc[:, 0], ay=self.acc[:, 1], beta=torch.atan2(s[:, 1], s[:, 0]),
                    steer=steer, omega=s[:, 3:7], x=x, y=y, yaw=yaw, fz=fz, torque=s[:, 7:11], tan_alpha=h[:, 0:4], kappa=h[:, 4:8])

    def build_library(self, *a, **k):
        self.lib = None
        return None

    def _place_v2(self, rows, v0, x0, y0, psi0, mu):
        n = rows.numel(); dt_ = self.dtype; dev = self.dev
        s = torch.zeros(n, 11, dtype=dt_, device=dev); s[:, 0] = v0; s[:, 3:7] = (v0 / self.ph.R)[:, None]
        u = torch.zeros(n, 8, dtype=dt_, device=dev)
        h = self.ph.init_hidden(s, u, mu.mean(1))
        self.s[rows] = s; self.h[rows] = h
        self.pose[rows] = torch.stack([x0 - P2.A_CG * torch.cos(psi0), y0 - P2.A_CG * torch.sin(psi0), psi0], 1)
        self.acc[rows] = 0.; self.kernel.bad[rows] = 0; self.torque_state[rows] = 0.; self.steer_state[rows] = 0.
        self.kernel.motor[rows] = 0.; self.kernel.steer[rows] = 0.

    def reset(self, rows=None, tid=None, s0=None, v0=None, mu=None, mu_hat=None):
        cfg = self.cfg; B = self.B; dev = self.dev; dt_ = self.dtype
        rows = torch.arange(B, device=dev) if rows is None else rows
        n = rows.numel()
        if n == 0:
            return
        U = lambda: torch.rand(n, generator=self.gen, device=dev, dtype=dt_)
        tid = torch.randint(0, self.pool.n_tracks, (n,), generator=self.gen, device=dev) if tid is None else tid
        if s0 is None:
            s0 = U() * self.pool.length[tid] if cfg.random_start else torch.zeros(n, dtype=dt_, device=dev)
        if v0 is None:
            v0 = cfg.v0_mps[0] + U() * (cfg.v0_mps[1] - cfg.v0_mps[0])
        if mu is None:
            mu = (cfg.mu_range[0] + U() * (cfg.mu_range[1] - cfg.mu_range[0]))[:, None].expand(n, 4)
        mu = mu.expand(n, 4) if mu.dim() == 2 else mu[:, None].expand(n, 4)
        if mu_hat is None:
            mu_hat = mu.mean(1) * (1 + cfg.mu_hat_rel_err * (2 * U() - 1))
        ip = torch.round(s0 / self.pool.ds).long() % self.pool.n[tid]
        x0 = self.pool.xy[tid, ip, 0]; y0 = self.pool.xy[tid, ip, 1]; psi0 = self.pool.psi[tid, ip]
        self._place_v2(rows, v0.to(dt_), x0, y0, psi0, mu.to(dt_))
        self.kernel.mu[rows] = mu.to(dt_); self.mu[rows] = mu.to(dt_); self.mu_hat[rows] = mu_hat.to(dt_)
        self.tid[rows] = tid; self.idx[rows] = ip
        self.t[rows] = 0.; self.slow_t[rows] = 0.; self.s_prog[rows] = 0.; self.laps[rows] = 0; self.prev_a[rows] = 0.
        self._frame(rows)
