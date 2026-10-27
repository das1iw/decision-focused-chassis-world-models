"""Batched GPU drift environment on the 44-state chassis world model (research/246, development).

The plant, actuator layer, signals and settled-start library are the 241 environment's (trfc_drift/chassis_env241_v1,
imported unchanged; the 239 fused kernel is the prospectively confirmed 235 driving plant). This file adds the DDEVEnv
drift task (trfc_drift/env.py, research/105-117):
- open courses (trfc_drift/tracks.random_course), which end at a finish line;
- reference sideslip beta_ref(s) = -sign(kappa) * 25 deg * clip((|kappa| - 1/80) / (1/80), 0, 1) (DDEVEnv.beta_ref);
- reward p * exp(-(e_lat / 2 m)^2) * [zone ? exp(-((beta - beta_ref) / 15 deg)^2) : 1] - 0.01 |da|^2, with progress
  p = ds / (dt * 20 m/s); -10 on crash (DDEVEnv rules), +10 at the course end;
- drift score of an episode D = completion x share of drift-zone steps with |beta - beta_ref| <= 5 deg (compare.py).
Observation: the 241 observation plus beta_ref * 2 at the same preview stations (as DDEVEnv's drift observation).
Training episodes start anywhere on the course (random_start) at a library speed; evaluation starts at s = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from trfc_drift.chassis_env241_v1 import ChassisEnv241, EnvConfig241, NACT, PRIV_DIM, obs_dim as obs_dim241

BETA_REF_DEG = 25.0
KAPPA_MIN = 1.0 / 80.0
ZONE_TOL_DEG = 5.0


@dataclass
class EnvConfig246(EnvConfig241):
    task: str = 'drift'
    t_max: float = 60.0
    v0_mps: tuple = (50 / 3.6, 80 / 3.6)          # DDEVEnv drift v0_kph (50, 80)
    lap_end: bool = True                          # the course end terminates (+10)
    finish_margin: float = 1.0                    # m before the course end (DDEVEnv)
    start_frac_max: float = 0.8                   # random training start within the first 80 % of the course
    beta_sigma_deg: float = 15.0                  # sideslip width of the drift reward (DDEVEnv: 15 deg); training option


def obs_dim(cfg):
    return obs_dim241(cfg) + len(cfg.preview_m)


class OpenTrackPool:
    """Open courses as padded GPU tensors with uniform station spacing ds (no wrap-around)."""

    def __init__(self, tracks, device, dtype):
        self.n_tracks = len(tracks); N = max(len(t.s) for t in tracks); self.ds = tracks[0].ds
        if any(abs(t.ds - self.ds) > 1e-12 or t.closed for t in tracks):
            raise ValueError('Open courses with equal ds required')
        xy = np.zeros((len(tracks), N, 2)); psi = np.zeros((len(tracks), N)); kap = np.zeros((len(tracks), N))
        for i, t in enumerate(tracks):
            n = len(t.s)
            if np.max(np.abs(np.diff(t.s) - t.ds)) > 1e-6:
                raise ValueError('Uniform station spacing required')
            xy[i, :n] = t.xy; psi[i, :n] = t.psi; kap[i, :n] = t.kappa
            xy[i, n:] = t.xy[-1]; psi[i, n:] = t.psi[-1]                    # padding repeats the last station
        T = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        self.xy, self.psi, self.kappa = T(xy), T(psi), T(kap)
        self.n = torch.as_tensor([len(t.s) for t in tracks], device=device)
        self.length = T([t.length for t in tracks]); self.names = [t.name for t in tracks]

    def interp(self, tid, s):
        """kappa, psi at stations s [B,K] on courses tid [B]; s is clamped to [0, length] (curvature 0 beyond the end)."""
        L = self.length[tid][:, None]; beyond = s > L
        s = torch.clamp(s, min=0.0); s = torch.minimum(s, L); n = self.n[tid][:, None]
        u = s / self.ds; i = torch.floor(u).long(); fr = u - i
        i0 = torch.minimum(i, n - 1); i1 = torch.minimum(i0 + 1, n - 1)
        fr = torch.where(i0 + 1 > n - 1, torch.zeros_like(fr), fr)
        rows = tid[:, None].expand_as(i0)
        k = self.kappa[rows, i0] * (1 - fr) + self.kappa[rows, i1] * fr
        p = self.psi[rows, i0] * (1 - fr) + self.psi[rows, i1] * fr
        return torch.where(beyond, torch.zeros_like(k), k), p

    def project(self, tid, x, y, hint, window=40):
        n = self.n[tid]; off = torch.arange(-window, window + 1, device=x.device)
        idx = torch.minimum(torch.clamp(hint[:, None] + off[None], min=0), (n - 1)[:, None]); rows = tid[:, None].expand_as(idx)
        d2 = (self.xy[rows, idx, 0] - x[:, None]) ** 2 + (self.xy[rows, idx, 1] - y[:, None]) ** 2
        j = idx.gather(1, d2.argmin(1, keepdim=True))[:, 0]
        ps = self.psi[tid, j]; tx, ty = torch.cos(ps), torch.sin(ps)
        dx = x - self.xy[tid, j, 0]; dy = y - self.xy[tid, j, 1]
        along = dx * tx + dy * ty; e_lat = -dx * ty + dy * tx
        s = torch.clamp(j.to(x.dtype) * self.ds + along, min=0.0)
        return j, e_lat, s


def beta_ref(kappa):
    z = torch.clamp((kappa.abs() - KAPPA_MIN) / KAPPA_MIN, 0., 1.)
    return -torch.sign(kappa) * math.radians(BETA_REF_DEG) * z


class DriftEnv246(ChassisEnv241):
    def __init__(self, plant, tracks, batch, cfg: EnvConfig246 | None = None, *, seed=0, device='cuda'):
        # ChassisEnv241.__init__ would build a closed-circuit TrackPool; the same members are built here with the open pool.
        self._init_open(plant, tracks, batch, cfg or EnvConfig246(), seed, device)

    def _init_open(self, plant, tracks, batch, cfg, seed, device):
        from trfc_drift.chassis_torch239_v1 import Chassis239
        from trfc_drift.chassis_cuda239_v1 import CudaChassis239
        self.cfg = cfg; self.B = B = int(batch); self.dev = torch.device(device); self.dtype = dt_ = cfg.dtype
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(int(seed))
        self.ch = Chassis239(plant, device=self.dev, dtype=dt_)
        self.kernel = CudaChassis239(self.ch, B, cfg.n_sub, dtype=dt_)
        self.pool = OpenTrackPool(tracks, self.dev, dt_)
        self.sm_cg = None
        self.m_tot = float(self.ch.mb + self.ch.mu_us.sum() + self.ch.ms.sum())
        z = lambda *s: torch.zeros(*s, dtype=dt_, device=self.dev)
        self.tid = torch.zeros(B, dtype=torch.long, device=self.dev); self.idx = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.s_tr = z(B); self.s_prog = z(B); self.t = z(B); self.slow_t = z(B); self.laps = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.prev_a = z(B, NACT); self.steer_state = z(B, 2); self.torque_state = z(B, 4); self.mu = z(B, 4); self.mu_hat = z(B)
        self.e_lat = z(B); self.e_psi = z(B); self.sig = {}
        self.preview = torch.as_tensor(cfg.preview_m, dtype=dt_, device=self.dev)
        self.side = self.ch.side
        self.zone_n = z(B); self.zone_ok = z(B); self.beta_ref_now = z(B)

    def reset(self, rows=None, tid=None, s0=None, v0=None, mu=None, mu_hat=None):
        cfg = self.cfg; dev = self.dev
        rows = torch.arange(self.B, device=dev) if rows is None else rows
        n = rows.numel()
        if n == 0:
            return
        if tid is None:
            tid = torch.randint(0, self.pool.n_tracks, (n,), generator=self.gen, device=dev)
        if s0 is None:
            if cfg.random_start:
                s0 = torch.rand(n, generator=self.gen, device=dev, dtype=self.dtype) * cfg.start_frac_max * self.pool.length[tid]
            else:
                s0 = torch.zeros(n, dtype=self.dtype, device=dev)
        super().reset(rows, tid=tid, s0=s0, v0=v0, mu=mu, mu_hat=mu_hat)
        self.zone_n[rows] = 0.; self.zone_ok[rows] = 0.
        k, _ = self.pool.interp(self.tid, self.s_tr[:, None]); self.beta_ref_now[rows] = beta_ref(k[:, 0])[rows]

    def obs(self):
        o = super().obs()
        k, _ = self.pool.interp(self.tid, self.s_tr[:, None] + self.preview[None])
        return torch.cat([o, beta_ref(k) * 2.0], -1)

    def step(self, action, auto_reset=True):
        cfg = self.cfg
        a = self.apply(action)
        self.kernel.replay()
        ds = self._frame()                                   # open course: no wrap
        self.s_prog += ds; self.t += cfg.dt
        da = a - self.prev_a; self.prev_a = a
        s = self.sig
        k, _ = self.pool.interp(self.tid, self.s_tr[:, None]); br = beta_ref(k[:, 0]); self.beta_ref_now = br
        in_zone = br.abs() > 1e-6
        q = torch.exp(-(self.e_lat / 2.0) ** 2)
        q = torch.where(in_zone, q * torch.exp(-((s['beta'] - br) / math.radians(cfg.beta_sigma_deg)) ** 2), q)
        r = ds / (cfg.dt * 20.) * q - 0.01 * (da * da).sum(-1)
        self.zone_n += in_zone.to(self.dtype)
        self.zone_ok += (in_zone & ((s['beta'] - br).abs() <= math.radians(ZONE_TOL_DEG))).to(self.dtype)
        bad = (self.kernel.bad > 0) | ~torch.isfinite(self.kernel.y).all(-1)
        off = self.e_lat.abs() > cfg.width + cfg.offtrack_margin
        spin = (s['beta'].abs() > math.radians(70.)) | (self.e_psi.abs() > math.radians(110.))
        self.slow_t = torch.where(s['vx'] < 2., self.slow_t + cfg.dt, torch.zeros_like(self.slow_t))
        stall = self.slow_t > 2.
        crash = bad | off | spin | stall
        finish = (self.s_tr >= self.pool.length[self.tid] - cfg.finish_margin) & ~crash
        r = torch.where(crash, r - 10., torch.where(finish, r + 10., r))
        r = torch.where(bad, torch.full_like(r, -10.), r)
        terminated = crash | finish
        truncated = (~terminated) & (self.t >= cfg.t_max - 1e-6)
        comp = torch.clamp(self.s_tr / self.pool.length[self.tid], 0., 1.)
        zfrac = torch.where(self.zone_n > 0, self.zone_ok / self.zone_n.clamp(min=1.), torch.zeros_like(self.zone_n))
        info = dict(crash=crash, bad=bad, off=off, spin=spin, stall=stall, finish=finish, progress=self.s_prog.clone(), t=self.t.clone(),
                    completion=comp, zone_frac=zfrac, D=comp * zfrac, in_zone=in_zone, beta=s['beta'].clone(), beta_ref=br.clone())
        info['final_obs'] = self.obs(); info['final_priv'] = self.priv()
        done = terminated | truncated
        if auto_reset and bool(done.any()):
            rows = done.nonzero()[:, 0]; self.reset(rows)
            return self.obs(), self.priv(), r, terminated, truncated, info
        return info['final_obs'], info['final_priv'], r, terminated, truncated, info


class FastDriftEnv246(DriftEnv246):
    """CUDA-graph version of DriftEnv246.step (the 241 fast-environment pattern, trfc_drift/chassis_env241_fast_v1):
    the reset-free part of a step is captured once on static buffers; resets of finished rows run eagerly."""
    SIG = ('vx', 'vy', 'r', 'ax', 'ay', 'beta', 'steer', 'omega', 'x', 'y', 'yaw', 'fz', 'torque', 'tan_alpha', 'kappa')

    def __init__(self, *a, graph=True, **k):
        super().__init__(*a, **k)
        self._graph = None; self._want_graph = graph
        self.act_buf = torch.zeros(self.B, NACT, dtype=self.dtype, device=self.dev)

    def _frame_inplace(self):
        sig = self.signals()
        if not self.sig:
            self.sig = {k: v.clone() for k, v in sig.items()}
        else:
            for k in self.SIG:
                self.sig[k].copy_(sig[k])
        j, e_lat, s = self.pool.project(self.tid, self.sig['x'], self.sig['y'], self.idx)
        _, psi = self.pool.interp(self.tid, s[:, None])
        e_psi = torch.atan2(torch.sin(self.sig['yaw'] - psi[:, 0]), torch.cos(self.sig['yaw'] - psi[:, 0]))
        ds = s - self.s_tr
        self.idx.copy_(j); self.e_lat.copy_(e_lat); self.e_psi.copy_(e_psi); self.s_tr.copy_(s)
        return ds

    def _signals_sub(self, rows):
        k = self.kernel

        class _K:
            pass
        sub = _K(); sub.y = k.y[rows]; sub.kp = k.kp[rows]; sub.steer = k.steer[rows]; sub.cache = k.cache[rows]
        sub.last = {key: v[rows] for key, v in k.last.items()}
        full = self.kernel; self.kernel = sub
        try:
            return self.signals()
        finally:
            self.kernel = full

    def _frame(self, rows=None):
        if rows is None:
            return self._frame_inplace()
        if not self.sig:
            self._frame_inplace(); return None
        sig = self._signals_sub(rows)
        for key in self.SIG:
            self.sig[key][rows] = sig[key]
        j, e_lat, s = self.pool.project(self.tid[rows], sig['x'], sig['y'], self.idx[rows])
        _, psi = self.pool.interp(self.tid[rows], s[:, None])
        e_psi = torch.atan2(torch.sin(sig['yaw'] - psi[:, 0]), torch.cos(sig['yaw'] - psi[:, 0]))
        self.idx[rows] = j; self.e_lat[rows] = e_lat; self.e_psi[rows] = e_psi; self.s_tr[rows] = s
        return None

    def _core(self):
        cfg = self.cfg; dt_ = self.dtype
        a = self.apply(self.act_buf)
        self.kernel.replay()
        ds = self._frame_inplace()
        self.s_prog.add_(ds); self.t.add_(cfg.dt)
        da = a - self.prev_a; self.prev_a.copy_(a)
        s = self.sig
        k, _ = self.pool.interp(self.tid, self.s_tr[:, None]); br = beta_ref(k[:, 0]); self.beta_ref_now.copy_(br)
        in_zone = br.abs() > 1e-6
        q = torch.exp(-(self.e_lat / 2.0) ** 2)
        q = torch.where(in_zone, q * torch.exp(-((s['beta'] - br) / math.radians(cfg.beta_sigma_deg)) ** 2), q)
        r = ds / (cfg.dt * 20.) * q - 0.01 * (da * da).sum(-1)
        self.zone_n.add_(in_zone.to(dt_))
        self.zone_ok.add_((in_zone & ((s['beta'] - br).abs() <= math.radians(ZONE_TOL_DEG))).to(dt_))
        bad = (self.kernel.bad > 0) | ~torch.isfinite(self.kernel.y).all(-1)
        off = self.e_lat.abs() > cfg.width + cfg.offtrack_margin
        spin = (s['beta'].abs() > math.radians(70.)) | (self.e_psi.abs() > math.radians(110.))
        self.slow_t.copy_(torch.where(s['vx'] < 2., self.slow_t + cfg.dt, torch.zeros_like(self.slow_t)))
        stall = self.slow_t > 2.
        crash = bad | off | spin | stall
        finish = (self.s_tr >= self.pool.length[self.tid] - cfg.finish_margin) & ~crash
        r = torch.where(crash, r - 10., torch.where(finish, r + 10., r))
        r = torch.where(bad, torch.full_like(r, -10.), r)
        terminated = crash | finish
        truncated = (~terminated) & (self.t >= cfg.t_max - 1e-6)
        comp = torch.clamp(self.s_tr / self.pool.length[self.tid], 0., 1.)
        zfrac = torch.where(self.zone_n > 0, self.zone_ok / self.zone_n.clamp(min=1.), torch.zeros_like(self.zone_n))
        out = dict(r=r, terminated=terminated, truncated=truncated, crash=crash, bad=bad, off=off, spin=spin, stall=stall, finish=finish,
                   progress=self.s_prog.clone(), t=self.t.clone(), completion=comp, zone_frac=zfrac, D=comp * zfrac, in_zone=in_zone,
                   beta=s['beta'].clone(), beta_ref=br.clone(), obs=self.obs(), priv=self.priv())
        if not hasattr(self, '_out'):
            self._out = {k_: v.clone() for k_, v in out.items()}
        else:
            for k_, v in out.items():
                self._out[k_].copy_(v)

    def _capture(self):
        state = [self.kernel.y, self.kernel.fp, self.kernel.cache, self.kernel.kp, self.kernel.bad, self.kernel.lastbuf, self.kernel.steer, self.kernel.motor,
                 self.steer_state, self.torque_state, self.prev_a, self.t, self.s_prog, self.slow_t, self.laps, self.idx, self.e_lat, self.e_psi, self.s_tr,
                 self.zone_n, self.zone_ok, self.beta_ref_now]
        state += [self.sig[k] for k in self.SIG]
        snap = [x.clone() for x in state]
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(2):
                self._core()
        torch.cuda.current_stream().wait_stream(st)
        for x, v in zip(state, snap):
            x.copy_(v)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph), torch.no_grad():
            self._core()
        for x, v in zip(state, snap):
            x.copy_(v)

    def step(self, action, auto_reset=True):
        if self._want_graph and self._graph is None:
            if not self.sig:
                self._frame_inplace()
            self._capture()
        self.act_buf.copy_(action)
        if self._graph is not None:
            self._graph.replay()
        else:
            with torch.no_grad():
                self._core()
        o = self._out
        info = {k: o[k] for k in ('crash', 'bad', 'off', 'spin', 'stall', 'finish', 'progress', 't', 'completion', 'zone_frac', 'D', 'in_zone',
                                  'beta', 'beta_ref')}
        info['final_obs'] = o['obs']; info['final_priv'] = o['priv']
        r, term, trunc = o['r'], o['terminated'], o['truncated']
        done = term | trunc
        if auto_reset and bool(done.any()):
            rows = done.nonzero()[:, 0]; self.reset(rows)
            return self.obs(), self.priv(), r, term, trunc, info
        return o['obs'], o['priv'], r, term, trunc, info


__all__ = ['EnvConfig246', 'DriftEnv246', 'FastDriftEnv246', 'OpenTrackPool', 'obs_dim', 'beta_ref', 'NACT', 'PRIV_DIM']
