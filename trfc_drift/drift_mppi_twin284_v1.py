"""GPU physics-v2 MPPI drift controller acting on the 44-state chassis world model (research/284, Track B).

The planner is planner v1/v3's drift MPPI (trfc_drift/mppi_numba.MPPINumba logic with the protocol-111 drift
settings: physics-v2 world model, n_sub 5, lambda 1, drift objective, sideslip barrier, envelope and lateral terminal
costs), with the sample rollout on the GPU (trfc_drift/mppi_cuda284_v1.py, parity 1e-10 with the CPU rollout) so that
K can be large and several episodes are planned at once. The plant is the 44-state CWM twin of CarSim
(trfc_drift/chassis_env246_v1, the 239 kernel), driven through its own actuator layer.

Observation path as the CarSim planner: physics-v2 state [vx, vy, r, omega x4, drive torque x4] from the plant's
CarSim-equivalent signals; the hidden lag/load states are initialized quasi-steadily and then updated by the teacher-
forced physics-v2 step with the applied commands (MPPINumba.observe). Friction input: the episode's mu_hat (probe)
or, with System 2, an external estimate.
"""
from __future__ import annotations

import math

import cupy as cp
import numpy as np
import torch

from trfc_drift import env as E_
from trfc_drift.mppi import envelope_vlim
from trfc_drift.mppi_cuda284_v1 import rollout_gpu
from trfc_drift.mppi_numba import MPPINumba, _step

DRIFT111 = dict(K=256, T=75, n_sub=5, lam=1.0, w_term=2.0, w_lat_term=0.5, v_margin=0.9, preview_m=50.0, w_spin=20.0,
                beta_safe_deg=40.0, beta_everywhere=True)


class TwinDriftMPPI:
    def __init__(self, E, K=256, T=75, seed=0, sigma=(0.25, 0.25, 0.35), colored=0.0, n_bias=0, sig_bias=0.3,
                 mufilter=None, precision='f32', **over):
        kw = dict(DRIFT111); kw.update(over); kw['K'] = K; kw['T'] = T
        self.cfg = E_.EnvConfig(task='drift', layout='4WS_TV', friction_info='estimate', probe=True)
        self.nb = MPPINumba(self.cfg, seed=seed, sigma=sigma, **kw)
        self.E, self.K, self.T, self.na = E, K, T, self.nb.na
        self.lam = self.nb.lam
        self.sig = cp.asarray(self.nb.sig)
        self.colored, self.n_bias, self.sig_bias = colored, n_bias, sig_bias
        self.rng = cp.random.default_rng(seed)
        self.ph = self.nb.ph
        self.mufilter = mufilter          # None or dict(factors=..., forget=..., window=10): System 2 (wm_mu_filter)
        self.precision = precision        # 'f32' (CUDA C kernel, mppi_cuda284_f32_v1) or 'f64' (numba-CUDA, parity kernel)
        if precision == 'f32':
            from trfc_drift.mppi_cuda284_f32_v1 import pack_device
            self.packed = pack_device(self.nb.MP)
        self.reset()

    def reset(self):
        self.Un = cp.zeros((self.E, self.T, self.na))
        self.hn = [None] * self.E
        self.sn_prev = [None] * self.E
        self.u_prev = [None] * self.E
        self.last = {}
        self.filters = None; self.mu_est = None; self.k_win = [0] * self.E
        self.est_log = []

    # ---------------------------------------------------------------- observation (CPU, per episode)
    def _signals(self, env):
        s = env.sig
        f = lambda t: t.detach().double().cpu().numpy()  # noqa: E731
        return dict(vx=f(s['vx']), vy=f(s['vy']), r=f(s['r']), omega=f(s['omega']), torque=f(s['torque']),
                    ax=f(s['ax']), ay=f(s['ay']), x=f(s['x']), y=f(s['y']), yaw=f(s['yaw']),
                    steer_cmd=f(env.kernel.steer), motor_cmd=f(env.kernel.motor), steer_state=f(env.steer_state),
                    torque_state=f(env.torque_state), prev_a=f(env.prev_a), idx=env.idx.cpu().numpy(),
                    tid=env.tid.cpu().numpy())

    def observe(self, sg, e, mu):
        s = np.concatenate([[sg['vx'][e], sg['vy'][e], sg['r'][e]], sg['omega'][e], sg['torque'][e]])
        u = np.concatenate([sg['steer_cmd'][e], sg['motor_cmd'][e]])
        if self.hn[e] is None:
            h = self.ph.init_hidden(torch.as_tensor(s)[None], torch.as_tensor(u)[None],
                                    torch.tensor([mu], dtype=torch.float64), torch.tensor([sg['ax'][e]]),
                                    torch.tensor([sg['ay'][e]]))
            self.hn[e] = h[0].numpy().astype(np.float64)
        else:
            # u is the command applied over the last step (sn_prev -> s), as MPPINumba.observe reads env.cmd
            _, self.hn[e] = _step(self.sn_prev[e], self.hn[e], u, mu, self.ph.n_sub, self.cfg.dt, *self.nb.MP)
        self.sn_prev[e] = s
        self.u_prev[e] = u
        if self.filters is not None:                       # System 2: physics-v2 replay friction filter (137)
            f = self.filters[e]
            if f.win is None:
                f.start_window(s, self.hn[e]); self.k_win[e] = 0
            else:
                f.add_command(u); self.k_win[e] += 1
                if self.k_win[e] >= self.mufilter.get('window', 10):
                    summ = f.update(s)
                    self.mu_est[e] = summ[self.mufilter.get('feed', 'mean')]          # 'mean' | 'q_lo' | 'map'
                    f.start_window(s, self.hn[e]); self.k_win[e] = 0

    def windows(self, env, sg, mus):
        c, T = self.cfg, self.T
        pool = env.pool
        ds = float(pool.ds)
        span = int((40.0 * T * c.dt + self.nb.preview_m + 20.0) / ds)
        E = self.E
        txy = np.zeros((E, span, 2)); tpsi = np.zeros((E, span)); tsw = np.zeros((E, span)); bref = np.zeros((E, span))
        vl = np.zeros((E, span)); nwin = np.zeros(E, np.int64); s0p = np.zeros(E)
        xy_all = pool.xy.detach().double().cpu().numpy(); psi_all = pool.psi.detach().double().cpu().numpy()
        kap_all = pool.kappa.detach().double().cpu().numpy(); n_all = pool.n.cpu().numpy()
        for e in range(E):
            t = int(sg['tid'][e]); n = int(n_all[t])
            rng_i = np.arange(int(sg['idx'][e]) - 10, int(sg['idx'][e]) - 10 + span)
            idx = np.clip(rng_i, 0, n - 1)
            sw = idx * ds
            kap = kap_all[t, idx]
            zz = np.clip((np.abs(kap) - c.drift_kappa_min) / c.drift_kappa_min, 0.0, 1.0)
            txy[e] = xy_all[t, idx]; tpsi[e] = psi_all[t, idx]; tsw[e] = sw
            bref[e] = -np.sign(kap) * np.radians(c.beta_ref_deg) * zz
            vl[e] = envelope_vlim(kap, float(mus[e]), self.nb.v_margin, ds, self.nb.preview_m)
            nwin[e] = span
            d0 = (txy[e, :, 0] - sg['x'][e]) ** 2 + (txy[e, :, 1] - sg['y'][e]) ** 2
            s0p[e] = sw[int(np.argmin(d0))]
        return txy, tpsi, tsw, bref, vl, nwin, s0p

    # ---------------------------------------------------------------- act
    def act(self, env, mu_hat, active=None):
        """mu_hat: numpy (E,) friction input per episode. Returns normalized actions torch (E, 6) on env device."""
        sg = self._signals(env)
        E, K, T, na = self.E, self.K, self.T, self.na
        mus = np.asarray(mu_hat, np.float64).copy()
        if self.mufilter is not None and self.filters is None:
            import types
            from trfc_drift.wm_mu_filter import MuFilter
            ep = np.zeros(15); ep[14] = self.ph.n_sub
            gym = types.SimpleNamespace(MP=self.nb.MP, EP=ep, cfg=self.cfg)
            fac = np.asarray(self.mufilter.get('factors', np.round(np.arange(0.60, 1.4001, 0.05), 3)))
            self.filters = [MuFilter(gym, float(mus[e]), factors=fac, forget=self.mufilter.get('forget', 0.97)) for e in range(E)]
            self.mu_est = mus.copy()
        if self.filters is not None:
            mus = self.mu_est.copy()
        for e in range(E):
            self.observe(sg, e, float(mus[e]))
        if self.filters is not None:
            mus = self.mu_est.copy()
            self.est_log.append(mus.copy())
        txy, tpsi, tsw, bref, vl, nwin, s0p = self.windows(env, sg, mus)
        eps = self.rng.standard_normal((E, K, T, na)) * self.sig
        if self.colored > 0:
            a = self.colored; b = math.sqrt(1 - a * a)
            for t in range(1, T):
                eps[:, :, t] = a * eps[:, :, t - 1] + b * eps[:, :, t]
        if self.n_bias > 0:
            nb = self.n_bias
            eps[:, 1:1 + nb] = 0.5 * eps[:, 1:1 + nb] + self.rng.standard_normal((E, nb, 1, na)) * self.sig_bias
        eps[:, 0] = 0.0
        A = cp.clip(self.Un[:, None] + eps, -1.0, 1.0)
        s0 = np.stack(self.sn_prev); h0 = np.stack(self.hn)
        pose0 = np.stack([sg['x'], sg['y'], sg['yaw']], 1)
        if self.precision == 'f32':
            from trfc_drift.mppi_cuda284_f32_v1 import rollout_gpu32
            G, alive = rollout_gpu32(A, s0, h0, sg['steer_state'], sg['torque_state'], pose0, sg['prev_a'], mus,
                                     self.ph.n_sub, self.nb.EP, self.packed, txy, tpsi, tsw, bref, vl, nwin, s0p)
            G = G.astype(cp.float64)
        else:
            G, alive = rollout_gpu(A, s0, h0, sg['steer_state'], sg['torque_state'], pose0, sg['prev_a'], mus,
                                   self.ph.n_sub, self.nb.EP, self.nb.MP, txy, tpsi, tsw, bref, vl, nwin, s0p)
        G = cp.asarray(G); alive = cp.asarray(alive)
        w = cp.exp((G - G.max(1, keepdims=True)) / self.lam)
        w /= w.sum(1, keepdims=True)
        U = (w[:, :, None, None] * A).sum(1)
        a0 = U[:, 0].copy()
        self.Un = cp.concatenate([U[:, 1:], U[:, -1:]], 1)
        self.last = dict(G_best=cp.asnumpy(G.max(1)), alive=cp.asnumpy(alive.mean(1)), ess=cp.asnumpy(1.0 / (w ** 2).sum(1)))
        return torch.as_tensor(a0, device=env.dev).to(env.dtype)


class MPPICuda284(MPPINumba):
    """Drop-in replacement of mppi_numba.MPPINumba for the CarSim harnesses (DDEVEnv): identical sampling, update and
    observer (MPPINumba.act / observe), with the sample rollout on the GPU (one episode). Same inputs as
    MPPINumba.rollout_G; returns host arrays (G, alive). precision 'f64' = numba-CUDA parity kernel (slow), 'f32' =
    CUDA C kernel (about 24 ms per 2048-sample plan)."""

    def __init__(self, env_cfg, precision='f32', **kw):
        super().__init__(env_cfg, **kw)
        self.precision = precision
        if precision == 'f32':
            from trfc_drift.mppi_cuda284_f32_v1 import pack_device
            self.packed = pack_device(self.MP)

    def rollout_G(self, env, A, mu_hat):
        c, T = self.c, self.T
        tr = env.track
        n = len(tr.s)
        span = int(min(n - 1 if tr.closed else n, (40.0 * T * c.dt + self.preview_m + 20.0) / tr.ds))
        rng_i = np.arange(env.idx - 10, env.idx - 10 + span)
        idx = rng_i % n if tr.closed else np.clip(rng_i, 0, n - 1)
        sw = tr.s[idx].astype(float)
        if tr.closed:
            sw = np.unwrap(sw * 2 * np.pi / tr.length) * tr.length / (2 * np.pi)
        kap = tr.kappa[idx]
        zz = np.clip((np.abs(kap) - c.drift_kappa_min) / c.drift_kappa_min, 0.0, 1.0)
        bref = -np.sign(kap) * np.radians(c.beta_ref_deg) * zz if c.task == "drift" else np.zeros_like(kap)
        vl = envelope_vlim(kap, mu_hat, self.v_margin, tr.ds, self.preview_m)
        d0 = (tr.xy[idx, 0] - env.state["x"]) ** 2 + (tr.xy[idx, 1] - env.state["y"]) ** 2
        s0 = float(sw[int(np.argmin(d0))])
        one = lambda x: np.asarray(x, np.float64)[None]  # noqa: E731
        if self.precision == 'f32':
            from trfc_drift.mppi_cuda284_f32_v1 import rollout_gpu32
            G, alive = rollout_gpu32(np.asarray(A)[None], one(self.sn_prev), one(self.hn), one(env.steer_state),
                                     one(env.torque_state), one([env.state["x"], env.state["y"], env.state["yaw"]]),
                                     one(env.prev_a), np.array([mu_hat], np.float64), self.ph.n_sub, self.EP, self.packed,
                                     one(tr.xy[idx]), one(tr.psi[idx]), one(sw), one(bref), one(vl),
                                     np.array([len(idx)], np.int64), np.array([s0], np.float64))
            return cp.asnumpy(G)[0].astype(np.float64), cp.asnumpy(alive)[0].astype(np.float64)
        G, alive = rollout_gpu(np.ascontiguousarray(A, dtype=np.float64)[None], one(self.sn_prev), one(self.hn),
                               one(env.steer_state), one(env.torque_state),
                               one([env.state["x"], env.state["y"], env.state["yaw"]]), one(env.prev_a),
                               np.array([mu_hat], np.float64), self.ph.n_sub, self.EP, self.MP,
                               one(tr.xy[idx]), one(tr.psi[idx]), one(sw), one(bref), one(vl),
                               np.array([len(idx)], np.int64), np.array([s0], np.float64))
        return G.copy_to_host()[0], alive.copy_to_host()[0]


class TwinSystem2:
    """System 2 for a policy on the 44-state twin: the planner's physics-v2 observer (hidden lag/load states, teacher-
    forced with the applied commands) and the protocol-137 friction filter per episode, without planning. update(env)
    is called once per control step before the policy acts and returns the friction estimate per episode (numpy E)."""

    def __init__(self, E, mu0, window=10, forget=0.97, feed='mean', factors=None):
        import types
        from trfc_drift.wm_mu_filter import MuFilter
        self.obs = TwinDriftMPPI(E, K=8, T=2, mufilter=None, precision='f64')     # observer only (no planning)
        ep = np.zeros(15); ep[14] = self.obs.ph.n_sub
        gym = types.SimpleNamespace(MP=self.obs.nb.MP, EP=ep, cfg=self.obs.cfg)
        fac = np.asarray(factors if factors is not None else np.round(np.arange(0.60, 1.4001, 0.05), 3))
        self.E, self.window, self.feed = E, window, feed
        self.filters = [MuFilter(gym, float(mu0[e]), factors=fac, forget=forget) for e in range(E)]
        self.mu_est = np.asarray(mu0, np.float64).copy()
        self.k_win = [0] * E

    def update(self, env):
        sg = self.obs._signals(env)
        for e in range(self.E):
            self.obs.observe(sg, e, float(self.mu_est[e]))
            s = self.obs.sn_prev[e]; u = self.obs.u_prev[e]; f = self.filters[e]
            if f.win is None:
                f.start_window(s, self.obs.hn[e]); self.k_win[e] = 0
            else:
                f.add_command(u); self.k_win[e] += 1
                if self.k_win[e] >= self.window:
                    self.mu_est[e] = f.update(s)[self.feed]
                    f.start_window(s, self.obs.hn[e]); self.k_win[e] = 0
        return self.mu_est.copy()
