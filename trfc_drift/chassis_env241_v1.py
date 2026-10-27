"""Batched GPU racing/drift environment on the 44-state chassis world model (research/241, development).

Plant: the accepted 239 fused kernel (trfc_drift/chassis_cuda239_v1, float32 by default), i.e. the prospectively
confirmed 235 driving plant; 20 ms control steps of 40 native 0.5 ms AB2 steps with commands held.

Task semantics follow DDEVEnv (trfc_drift/env.py), layout 4WS_TV, action a in [-1,1]^6 = [df, dr, T_FL, T_FR, T_RL, T_RR]:
- steer targets df*30 deg (both front wheels), dr*10 deg (both rear wheels), rate-limited 90/45 deg/s;
- wheel torque targets a*2000 N m, rate-limited 20000 N m/s, drive taper to zero between omega_lim-band and omega_lim
  (150 rad/s), regen taper to zero near standstill (DDEVEnv._limit);
- NEW (shared with the CarSim adapter): a total electric power budget. Drive power sum(max(T*omega, 0)) and regen power
  sum(max(-T*omega, 0)) are each capped at power_kw by scaling those torques. CarSim's battery limit (about 190 kW)
  is not in the 235 plant, so the budget keeps it from binding.
- race reward: progress ds/(dt*20 m/s) - 0.01|da|^2; -10 on crash (off track |e_lat| > w + 1.5 m, |beta| > 70 deg or
  |e_psi| > 110 deg, vx < 2 m/s for 2 s, plant flag/nonfinite). Training episodes run a fixed time without lap end
  (wm_gym note: lap termination taught slower driving); evaluation ends at one lap.
Observation signals reproduce CarSim's exports (research/07, 2026-10-05 18:58):
- Vx/Vy: total-CG velocity in the yaw frame (momentum / mass);
- Ax/Ay: total-CG acceleration in the yaw frame without gravity, from the tire and aero forces;
- Beta: atan2(Vy, Vx);
- Steer: the composed road-wheel angle (mirrored toe + compliance + imported steer);
- AVy: the relative wheel spin;
- Xo/Yo: the vehicle origin.
Actor observation (OBS): motion [vx/20, vy/5, r, ax/10, ay/10, beta], wheel-speed ratios omega*0.287/vx - 1
(clipped), realized front/rear steer*3, previous action, torque state/2000, [e_lat/w, sin e_psi, cos e_psi], curvature
preview*20 (clipped) at PREVIEW stations, and mu_hat (true mu times a per-episode relative error). Privileged critic
extras (PRIV): true per-wheel mu, tan(alpha), kappa, Fz/5000, realized motor torque/2000, time fraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from trfc_drift.chassis_torch239_v1 import Chassis239, _mm, _mv
from trfc_drift.chassis_cuda239_v1 import CudaChassis239

R_WHEEL = 0.287                                  # carsim_dll.VEH['r_wheel'], as in DDEVEnv's observation
PREVIEW = (0., 5., 10., 15., 20., 30., 40., 55., 70., 90., 115., 145.)


@dataclass
class EnvConfig241:
    task: str = 'race'
    dt: float = 0.02
    n_sub: int = 40
    t_max: float = 40.0                          # training episode length, s (fixed time, no lap end)
    lap_end: bool = False
    v0_mps: tuple = (8.0, 25.0)
    mu_range: tuple = (0.3, 1.0)
    mu_hat_rel_err: float = 0.05                 # uniform relative error of mu_hat per episode
    steer_f_max_deg: float = 30.0
    steer_r_max_deg: float = 10.0
    steer_rate_f_dps: float = 90.0
    steer_rate_r_dps: float = 45.0
    torque_max: float = 2000.0
    torque_rate: float = 20000.0
    omega_lim: float = 150.0
    omega_band: float = 10.0
    power_kw: float = 150.0
    width: float = 5.0
    offtrack_margin: float = 1.5
    preview_m: tuple = PREVIEW
    random_start: bool = True
    edge_weight: float = 0.0                     # training shaping: -w * relu(|e_lat| - (width - edge_margin))^2 per step
    edge_margin: float = 1.0
    dtype: torch.dtype = torch.float32


NACT = 6


def obs_dim(cfg):
    return 6 + 4 + 2 + NACT + 4 + 3 + len(cfg.preview_m) + 1


PRIV_DIM = 4 + 4 + 4 + 4 + 4 + 1


class TrackPool:
    """Closed circuits (trfc_drift.circuits.random_circuit) as padded GPU tensors with uniform station spacing ds."""

    def __init__(self, tracks, device, dtype):
        self.n_tracks = len(tracks); N = max(len(t.s) for t in tracks); self.ds = tracks[0].ds
        if any(abs(t.ds - self.ds) > 1e-12 or not t.closed for t in tracks):
            raise ValueError('Closed tracks with equal ds required')
        xy = np.zeros((len(tracks), N, 2)); psi = np.zeros((len(tracks), N)); kap = np.zeros((len(tracks), N))
        for i, t in enumerate(tracks):
            n = len(t.s)
            if np.max(np.abs(np.diff(t.s) - t.ds)) > 1e-6:
                raise ValueError('Uniform station spacing required')
            xy[i, :n] = t.xy; psi[i, :n] = t.psi; kap[i, :n] = t.kappa
        T = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        self.xy, self.psi, self.kappa = T(xy), T(psi), T(kap)
        self.n = torch.as_tensor([len(t.s) for t in tracks], device=device)
        self.length = T([t.length for t in tracks]); self.names = [t.name for t in tracks]

    def interp(self, tid, s):
        """kappa, psi at stations s [B,K] (wrapped) on tracks tid [B]."""
        L = self.length[tid][:, None]; s = torch.remainder(s, L); n = self.n[tid][:, None]
        u = s / self.ds; i = torch.floor(u).long(); fr = u - i
        i0 = torch.minimum(i, n - 1); i1 = torch.where(i0 + 1 >= n, n - 1, i0 + 1)
        fr = torch.where(i0 + 1 >= n, torch.zeros_like(fr), fr)
        rows = tid[:, None].expand_as(i0)
        k = self.kappa[rows, i0] * (1 - fr) + self.kappa[rows, i1] * fr
        p = self.psi[rows, i0] * (1 - fr) + self.psi[rows, i1] * fr
        return k, p

    def project(self, tid, x, y, hint, window=40):
        n = self.n[tid]; off = torch.arange(-window, window + 1, device=x.device)
        idx = torch.remainder(hint[:, None] + off[None], n[:, None]); rows = tid[:, None].expand_as(idx)
        d2 = (self.xy[rows, idx, 0] - x[:, None]) ** 2 + (self.xy[rows, idx, 1] - y[:, None]) ** 2
        j = idx.gather(1, d2.argmin(1, keepdim=True))[:, 0]
        ps = self.psi[tid, j]; tx, ty = torch.cos(ps), torch.sin(ps)
        dx = x - self.xy[tid, j, 0]; dy = y - self.xy[tid, j, 1]
        along = dx * tx + dy * ty; e_lat = -dx * ty + dy * tx
        s = torch.remainder(j.to(x.dtype) * self.ds + along, self.length[tid])
        return j, e_lat, s


class ChassisEnv241:
    def __init__(self, plant, tracks, batch, cfg: EnvConfig241 | None = None, *, seed=0, device='cuda', settle_s=1.0):
        self.cfg = cfg = cfg or EnvConfig241(); self.B = B = int(batch); self.dev = torch.device(device)
        self.dtype = cfg.dtype; dt_ = self.dtype
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(int(seed))
        self.ch = Chassis239(plant, device=self.dev, dtype=dt_)
        self.kernel = CudaChassis239(self.ch, B, cfg.n_sub, dtype=dt_)
        self.pool = TrackPool(tracks, self.dev, dt_)
        self.sm_cg = None                         # set_origin_offset() before reset
        self.m_tot = float(self.ch.mb + self.ch.mu_us.sum() + self.ch.ms.sum())
        z = lambda *s: torch.zeros(*s, dtype=dt_, device=self.dev)
        self.tid = torch.zeros(B, dtype=torch.long, device=self.dev); self.idx = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.s_tr = z(B); self.s_prog = z(B); self.t = z(B); self.slow_t = z(B); self.laps = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.prev_a = z(B, NACT); self.steer_state = z(B, 2); self.torque_state = z(B, 4); self.mu = z(B, 4); self.mu_hat = z(B)
        self.e_lat = z(B); self.e_psi = z(B); self.sig = {}
        self.preview = torch.as_tensor(cfg.preview_m, dtype=dt_, device=self.dev)
        self.side = self.ch.side

    # ------------------------------------------------------------------ settled initial states
    def set_origin_offset(self, sm_cg):
        """Sprung CG in vehicle (origin) coordinates [x, y, z] m: X_CG = Xo + R sm_cg."""
        self.sm_cg = torch.as_tensor(np.asarray(sm_cg, float), dtype=self.dtype, device=self.dev)

    def build_library(self, base, speeds, settle_s=1.0, mu=0.9, kv=400.):
        """Settled straight-running states at the given speeds from one native base state.
        base: dict y [44], fp [44], cache [4,3], kp. Speeds scaled on v[0:3] and the relative spins, then settled with a
        speed-holding torque on all wheels; returns tensors stored as the library."""
        n = len(speeds); dt_ = self.dtype; dev = self.dev
        k = CudaChassis239(self.ch, n, self.cfg.n_sub, dtype=dt_)
        T = lambda x: torch.as_tensor(np.asarray(x, float), dtype=dt_, device=dev)
        y = T(base['y'])[None].repeat(n, 1); v_base = float(torch.linalg.norm(y[0, 14:16]))
        sc = T(speeds) / v_base
        y[:, 14:17] *= sc[:, None]; y[:, 24:28] *= sc[:, None]; y[:, 32:40] *= 0.; y[:, 40:44] *= 0.
        k.y.copy_(y); k.fp.zero_(); k.fp[:, :14] = y[:, 14:28]
        k.cache.copy_(T(base['cache'])[None].repeat(n, 1, 1)); k.kp.fill_(float(base['kp'])); k.mu.fill_(mu)
        target = T(speeds)
        for _ in range(int(round(settle_s / self.cfg.dt))):
            v = torch.linalg.norm(k.y[:, 14:16], dim=1)
            k.motor.copy_(torch.clamp(kv * (target - v), -600., 600.)[:, None].expand(n, 4))
            k.replay()
        if bool((k.bad > 0).any()):
            raise FloatingPointError('Library settling flagged')
        self.lib = dict(y=k.y.clone(), fp=k.fp.clone(), cache=k.cache.clone(), kp=k.kp.clone(), torque=k.y[:, 40:44].clone(),
                        speed=torch.linalg.norm(k.y[:, 14:16], dim=1))
        return self.lib

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _rotz(vec, c, s):
        x, y = vec[..., 0], vec[..., 1]
        return torch.stack([c * x - s * y, s * x + c * y], -1)

    def _place(self, rows, lib_i, x0, y0, psi0):
        """Write library states lib_i into kernel rows with the vehicle origin at (x0, y0) and heading psi0."""
        L = self.lib; k = self.kernel
        y = L['y'][lib_i].clone(); fp = L['fp'][lib_i].clone()
        dpsi = psi0 - y[:, 5]; c, s = torch.cos(dpsi), torch.sin(dpsi)
        y[:, 5] = psi0
        y[:, 14:16] = self._rotz(y[:, 14:16], c, s); fp[:, 0:2] = self._rotz(fp[:, 0:2], c, s); fp[:, 14:16] = self._rotz(fp[:, 14:16], c, s)
        R = self._R(y[:, 3:6])
        off = _mv(R, self.sm_cg.expand(len(rows), 3))
        y[:, 0] = x0 + off[:, 0]; y[:, 1] = y0 + off[:, 1]
        k.y[rows] = y; k.fp[rows] = fp; k.cache[rows] = L['cache'][lib_i]; k.kp[rows] = L['kp'][lib_i]; k.bad[rows] = 0
        self.torque_state[rows] = L['torque'][lib_i]; self.steer_state[rows] = 0.
        k.motor[rows] = L['torque'][lib_i]; k.steer[rows] = 0.; k.lastbuf[rows] = 0.   # stale last-step forces cleared

    @staticmethod
    def _R(eul):
        from trfc_drift.chassis_torch239_v1 import _rot
        Rx, _ = _rot(eul[:, 0], 0); Ry, _ = _rot(eul[:, 1], 1); Rz, _ = _rot(eul[:, 2], 2)
        return _mm(_mm(Rz, Ry), Rx)

    def signals(self):
        """CarSim-equivalent signals from the kernel state and held values (see module docstring)."""
        k = self.kernel; y = k.y; q, v = y[:, :14], y[:, 14:28]
        system = self.ch.steer_comp * k.kp
        extra = k.steer + torch.stack([system, system, torch.zeros_like(system), torch.zeros_like(system)], -1)
        R, W, Ju, Js, u, s_, Q, A = self.ch.geometry(q, k.cache, extra)
        mom = self.ch.mb * v[:, :3] + (self.ch.mu_us[None, :, None] * _mv(Ju, v[:, None])).sum(1) + (self.ch.ms[None, :, None] * _mv(Js, v[:, None])).sum(1)
        vcg = mom / self.m_tot; cy, sy = torch.cos(q[:, 5]), torch.sin(q[:, 5])
        vx = cy * vcg[:, 0] + sy * vcg[:, 1]; vy = -sy * vcg[:, 0] + cy * vcg[:, 1]
        r = (W[:, 2, :] * v[:, 3:6]).sum(-1)
        C = _mm(R[:, None], Q); spin = C[..., 1]
        gx = torch.stack([spin[..., 1], -spin[..., 0]], -1); gx = gx / torch.linalg.norm(gx, dim=-1, keepdim=True)
        gy = torch.stack([-gx[..., 1], gx[..., 0]], -1)
        last = k.last
        F = (last['fx'][..., None] * gx + last['fy'][..., None] * gy).sum(1)
        Fa, _ = self.ch.aero_wrench(R, v); Fa_w = _mv(R, Fa)[:, :2]
        a = (F + Fa_w) / self.m_tot
        ax = cy * a[:, 0] + sy * a[:, 1]; ay = -sy * a[:, 0] + cy * a[:, 1]
        val, _ = self.ch._curves(q[:, 6:10]); dsteer, _ = self.ch.compliance(k.cache, extra)
        steer = -self.side * val[4] + dsteer
        origin = q[:, :3] - _mv(R, self.sm_cg.expand(q.shape[0], 3))
        return dict(vx=vx, vy=vy, r=r, ax=ax, ay=ay, beta=torch.atan2(vy, vx), steer=steer, omega=v[:, 10:14],
                    x=origin[:, 0], y=origin[:, 1], yaw=q[:, 5], fz=last['fz'], torque=y[:, 40:44],
                    tan_alpha=y[:, 32:36], kappa=y[:, 36:40])

    # ------------------------------------------------------------------ episode
    def reset(self, rows=None, tid=None, s0=None, v0=None, mu=None, mu_hat=None):
        """Reset rows (all by default). Unspecified tracks, start stations, speeds and friction are sampled."""
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
        sp = self.lib['speed']; hi = torch.bucketize(v0, sp).clamp(1, sp.numel() - 1)       # nearest library speed
        lib_i = torch.where((v0 - sp[hi - 1]).abs() <= (sp[hi] - v0).abs(), hi - 1, hi)
        ip = torch.round(s0 / self.pool.ds).long() % self.pool.n[tid]
        x0 = self.pool.xy[tid, ip, 0]; y0 = self.pool.xy[tid, ip, 1]; psi0 = self.pool.psi[tid, ip]
        self._place(rows, lib_i, x0, y0, psi0)
        self.kernel.mu[rows] = mu.to(dt_); self.mu[rows] = mu.to(dt_); self.mu_hat[rows] = mu_hat.to(dt_)
        self.tid[rows] = tid; self.idx[rows] = ip
        self.t[rows] = 0.; self.slow_t[rows] = 0.; self.s_prog[rows] = 0.; self.laps[rows] = 0; self.prev_a[rows] = 0.
        self.kernel.brake[rows] = 0.; self.kernel.external[rows] = 0.
        self._frame(rows)

    def _frame(self, rows=None):
        sig = self.signals(); self.sig = sig
        j, e_lat, s = self.pool.project(self.tid, sig['x'], sig['y'], self.idx)
        _, psi = self.pool.interp(self.tid, s[:, None])
        e_psi = torch.atan2(torch.sin(sig['yaw'] - psi[:, 0]), torch.cos(sig['yaw'] - psi[:, 0]))
        if rows is None:
            self.idx, self.e_lat, self.e_psi = j, e_lat, e_psi; ds = s - self.s_tr; self.s_tr = s
            return ds
        self.idx[rows] = j[rows]; self.e_lat[rows] = e_lat[rows]; self.e_psi[rows] = e_psi[rows]; self.s_tr[rows] = s[rows]
        return None

    def obs(self):
        cfg = self.cfg; s = self.sig
        vx = torch.clamp(s['vx'], min=1.0)
        ratio = torch.clamp(s['omega'] * R_WHEEL / vx[:, None] - 1.0, -1.0, 2.0)
        k, _ = self.pool.interp(self.tid, self.s_tr[:, None] + self.preview[None])
        parts = [torch.stack([s['vx'] / 20., s['vy'] / 5., s['r'], s['ax'] / 10., s['ay'] / 10., s['beta']], -1), ratio,
                 torch.stack([.5 * (s['steer'][:, 0] + s['steer'][:, 1]), .5 * (s['steer'][:, 2] + s['steer'][:, 3])], -1) * 3.,
                 self.prev_a, self.torque_state / cfg.torque_max,
                 torch.stack([self.e_lat / cfg.width, torch.sin(self.e_psi), torch.cos(self.e_psi)], -1), torch.clamp(k * 20., -2., 2.),
                 self.mu_hat[:, None]]
        o = torch.cat(parts, -1)
        return torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)

    def priv(self):
        s = self.sig
        p = torch.cat([self.mu, s['tan_alpha'], s['kappa'], s['fz'] / 5000., s['torque'] / self.cfg.torque_max, (self.t / self.cfg.t_max)[:, None]], -1)
        return torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------ actuator layer
    def apply(self, a):
        cfg = self.cfg; k = self.kernel
        a = torch.clamp(a, -1., 1.)
        df = a[:, 0] * math.radians(cfg.steer_f_max_deg); dr = a[:, 1] * math.radians(cfg.steer_r_max_deg)
        rf = math.radians(cfg.steer_rate_f_dps) * cfg.dt; rr = math.radians(cfg.steer_rate_r_dps) * cfg.dt
        st = self.steer_state
        st[:, 0] += torch.clamp(df - st[:, 0], -rf, rf); st[:, 1] += torch.clamp(dr - st[:, 1], -rr, rr)
        k.steer.copy_(torch.stack([st[:, 0], st[:, 0], st[:, 1], st[:, 1]], -1))
        rt = cfg.torque_rate * cfg.dt
        self.torque_state += torch.clamp(a[:, 2:] * cfg.torque_max - self.torque_state, -rt, rt)
        T4 = self.torque_state.clone(); om = self.sig['omega']
        T4 = torch.where(T4 > 0, T4 * torch.clamp((cfg.omega_lim - om) / cfg.omega_band, 0., 1.), T4)
        T4 = torch.where(T4 < 0, T4 * torch.clamp(om / cfg.omega_band, 0., 1.), T4)
        P = T4 * om; pmax = cfg.power_kw * 1000.
        Pd = torch.clamp(P, min=0.).sum(1, keepdim=True); Pr = torch.clamp(-P, min=0.).sum(1, keepdim=True)
        sd = torch.where(Pd > pmax, pmax / Pd.clamp(min=1e-6), torch.ones_like(Pd)); sr = torch.where(Pr > pmax, pmax / Pr.clamp(min=1e-6), torch.ones_like(Pr))
        T4 = torch.where(P > 0, T4 * sd, torch.where(P < 0, T4 * sr, T4))
        k.motor.copy_(T4)
        return a

    # ------------------------------------------------------------------ step
    def step(self, action, auto_reset=True):
        cfg = self.cfg
        a = self.apply(action)
        self.kernel.replay()
        ds = self._frame()
        Lt = self.pool.length[self.tid]
        wrap_f = ds < -0.5 * Lt; wrap_b = ds > 0.5 * Lt
        ds = torch.where(wrap_f, ds + Lt, torch.where(wrap_b, ds - Lt, ds)); self.laps += wrap_f.long()
        self.s_prog += ds; self.t += cfg.dt
        da = a - self.prev_a; self.prev_a = a
        p = ds / (cfg.dt * 20.)
        r = p - 0.01 * (da * da).sum(-1)
        if cfg.edge_weight > 0:
            r = r - cfg.edge_weight * torch.clamp(self.e_lat.abs() - (cfg.width - cfg.edge_margin), min=0.) ** 2
        s = self.sig
        bad = (self.kernel.bad > 0) | ~torch.isfinite(self.kernel.y).all(-1)
        off = self.e_lat.abs() > cfg.width + cfg.offtrack_margin
        spin = (s['beta'].abs() > math.radians(70.)) | (self.e_psi.abs() > math.radians(110.))
        self.slow_t = torch.where(s['vx'] < 2., self.slow_t + cfg.dt, torch.zeros_like(self.slow_t))
        stall = self.slow_t > 2.
        crash = bad | off | spin | stall
        finish = (self.laps >= 1) & torch.tensor(cfg.lap_end, device=self.dev) & ~crash
        r = torch.where(crash, r - 10., torch.where(finish, r + 10., r))
        r = torch.where(bad, torch.full_like(r, -10.), r)
        terminated = crash | finish
        truncated = (~terminated) & (self.t >= cfg.t_max - 1e-6)
        info = dict(crash=crash, bad=bad, off=off, spin=spin, stall=stall, finish=finish, progress=self.s_prog.clone(), t=self.t.clone(), laps=self.laps.clone())
        info['final_obs'] = self.obs(); info['final_priv'] = self.priv()      # pre-reset (bootstrap on truncation)
        done = terminated | truncated
        if auto_reset and bool(done.any()):
            rows = done.nonzero()[:, 0]; self.reset(rows)
            return self.obs(), self.priv(), r, terminated, truncated, info
        return info['final_obs'], info['final_priv'], r, terminated, truncated, info
