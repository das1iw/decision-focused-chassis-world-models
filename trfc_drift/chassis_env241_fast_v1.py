"""CUDA-graph version of the 241 environment step (research/241, development; same semantics as ChassisEnv241.step).

The reset-free part of a step (actuator layer, kernel launch, signals, track frame, reward, flags and observations)
is captured once as a CUDA graph on static buffers; resets of finished rows run eagerly on those rows only, with the
signals recomputed for the reset rows alone. Results equal ChassisEnv241.step for the same actions (tested).
"""
from __future__ import annotations

import math

import torch

from trfc_drift.chassis_env241_v1 import ChassisEnv241


class FastChassisEnv241(ChassisEnv241):
    SIG = ('vx', 'vy', 'r', 'ax', 'ay', 'beta', 'steer', 'omega', 'x', 'y', 'yaw', 'fz', 'torque', 'tan_alpha', 'kappa')

    def __init__(self, *a, graph=True, **k):
        super().__init__(*a, **k)
        self._graph = None; self._want_graph = graph
        B, dev, dt_ = self.B, self.dev, self.dtype
        self.act_buf = torch.zeros(B, 6, dtype=dt_, device=dev)

    # ------------------------------------------------------------------ in-place frame
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

    def _frame_rows(self, rows):
        """Signals and frame for the given rows only (after a reset)."""
        k = self.kernel; q = k.y[rows]
        sig = self._signals_sub(rows)
        for key in self.SIG:
            self.sig[key][rows] = sig[key]
        j, e_lat, s = self.pool.project(self.tid[rows], sig['x'], sig['y'], self.idx[rows])
        _, psi = self.pool.interp(self.tid[rows], s[:, None])
        e_psi = torch.atan2(torch.sin(sig['yaw'] - psi[:, 0]), torch.cos(sig['yaw'] - psi[:, 0]))
        self.idx[rows] = j; self.e_lat[rows] = e_lat; self.e_psi[rows] = e_psi; self.s_tr[rows] = s

    def _signals_sub(self, rows):
        k = self.kernel
        class _K:   # row view of the kernel buffers
            pass
        sub = _K(); sub.y = k.y[rows]; sub.kp = k.kp[rows]; sub.steer = k.steer[rows]; sub.cache = k.cache[rows]
        L = k.last; sub.last = {key: v[rows] for key, v in L.items()}
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
        self._frame_rows(rows); return None

    # ------------------------------------------------------------------ captured core
    def _core(self):
        cfg = self.cfg
        a = self.apply(self.act_buf)
        self.kernel.replay()
        ds = self._frame_inplace()
        Lt = self.pool.length[self.tid]
        wrap_f = ds < -0.5 * Lt; wrap_b = ds > 0.5 * Lt
        ds = torch.where(wrap_f, ds + Lt, torch.where(wrap_b, ds - Lt, ds)); self.laps.add_(wrap_f.long())
        self.s_prog.add_(ds); self.t.add_(cfg.dt)
        da = a - self.prev_a; self.prev_a.copy_(a)
        r = ds / (cfg.dt * 20.) - 0.01 * (da * da).sum(-1)
        if cfg.edge_weight > 0:
            r = r - cfg.edge_weight * torch.clamp(self.e_lat.abs() - (cfg.width - cfg.edge_margin), min=0.) ** 2
        s = self.sig
        bad = (self.kernel.bad > 0) | ~torch.isfinite(self.kernel.y).all(-1)
        off = self.e_lat.abs() > cfg.width + cfg.offtrack_margin
        spin = (s['beta'].abs() > math.radians(70.)) | (self.e_psi.abs() > math.radians(110.))
        self.slow_t.copy_(torch.where(s['vx'] < 2., self.slow_t + cfg.dt, torch.zeros_like(self.slow_t)))
        stall = self.slow_t > 2.
        crash = bad | off | spin | stall
        finish = (self.laps >= 1) & (~crash) if cfg.lap_end else torch.zeros_like(crash)
        r = torch.where(crash, r - 10., torch.where(finish, r + 10., r))
        r = torch.where(bad, torch.full_like(r, -10.), r)
        terminated = crash | finish
        truncated = (~terminated) & (self.t >= cfg.t_max - 1e-6)
        out = dict(r=r, terminated=terminated, truncated=truncated, crash=crash, bad=bad, off=off, spin=spin, stall=stall, finish=finish,
                   progress=self.s_prog.clone(), t=self.t.clone(), laps=self.laps.clone(), obs=self.obs(), priv=self.priv())
        if not hasattr(self, '_out'):
            self._out = {k: v.clone() for k, v in out.items()}
        else:
            for k, v in out.items():
                self._out[k].copy_(v)

    def _capture(self):
        state = [self.kernel.y, self.kernel.fp, self.kernel.cache, self.kernel.kp, self.kernel.bad, self.kernel.lastbuf, self.kernel.steer, self.kernel.motor,
                 self.steer_state, self.torque_state, self.prev_a, self.t, self.s_prog, self.slow_t, self.laps, self.idx, self.e_lat, self.e_psi, self.s_tr]
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
        info = dict(crash=o['crash'], bad=o['bad'], off=o['off'], spin=o['spin'], stall=o['stall'], finish=o['finish'], progress=o['progress'],
                    t=o['t'], laps=o['laps'], final_obs=o['obs'], final_priv=o['priv'])
        r, term, trunc = o['r'], o['terminated'], o['truncated']
        done = term | trunc
        if auto_reset and bool(done.any()):
            rows = done.nonzero()[:, 0]; self.reset(rows)
            return self.obs(), self.priv(), r, term, trunc, info
        return o['obs'], o['priv'], r, term, trunc, info
