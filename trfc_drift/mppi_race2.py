"""Race planner v2 (after protocol 112): MPPINumba with a mixture of sample proposals.

Diagnosis (research/07, 2026-09-26; trfc_drift/diag_race_replay.py on the 112 run 4WS_TV seed 3, race_B mu 0.65):
at the corner entry the warm-started mean plan steered about -10 deg (right) into a left-hander. A pure-pursuit plan
scored higher under the planner's own objective than all 256 samples at every step from 9.0 to 9.7 s, but white
per-step noise around the mean never reached it; after 9.8 s every sample was predicted to leave the track.

v2 keeps the objective, the physics-v2 world model and the compiled rollout (mppi_numba.rollout) unchanged. It
changes only where the K samples come from:
  - K - K_b - K_n samples: warm start + white noise (as v1; sample 0 noise-free);
  - K_b samples: warm start + bias_white x white noise + a per-sample offset held over the whole horizon
    (N(0, sig_bias) per action), which searches low-frequency changes such as "steer further left";
  - K_n samples: a closed-loop nominal plan + white noise (its first sample noise-free). The nominal is simulated
    each step with the same world model and actuator limits: pure-pursuit front steer (lookahead max(ld_min, ld_t v)),
    no rear steer, and equal wheel torques tracking the planner's envelope speed (gain k_v, limits a_min..a_max).
With K_b = K_n = 0 it reproduces MPPINumba bit for bit (check_v1_equivalence).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from trfc_drift.mppi import envelope_vlim
from trfc_drift.mppi_numba import MPPINumba, _step, rollout
from trfc_drift.wm_physics2 import L_WB, M


@njit(cache=True)
def nominal_plan(T, na, s0, h0, st0, ts0, pose0, mu, n_sub, eP, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy,
                 FY, am, fzm, MZ, fzg, kpk_t, tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt, txy, tpsi, vl, ds, wheelbase,
                 mass, r0, ld_min, ld_t, k_v, a_min, a_max):
    """Closed-loop pure-pursuit + envelope-speed plan (T, na) in normalized actions, simulated with the world model."""
    dt = eP[0]
    layout = int(eP[9])
    four = layout == 0 or layout == 4
    A = np.zeros((T, na))
    s = s0.copy()
    h = h0.copy()
    sf, sr = st0[0], st0[1]
    ts = ts0.copy()
    x, y, yaw = pose0[0], pose0[1], pose0[2]
    nw = txy.shape[0]
    j = 0
    best = 1e30
    for w in range(nw):
        d2 = (x - txy[w, 0]) ** 2 + (y - txy[w, 1]) ** 2
        if d2 < best:
            best = d2
            j = w
    n_drv = 2.0 if (layout == 1 or layout == 4) else 4.0
    u = np.empty(8)
    for t in range(T):
        v = math.sqrt(s[0] * s[0] + s[1] * s[1])
        ld = max(ld_min, ld_t * v)
        jt = min(j + int(ld / ds), nw - 1)
        ang = math.atan2(txy[jt, 1] - y, txy[jt, 0] - x) - yaw
        ang = math.atan2(math.sin(ang), math.cos(ang))
        delta = math.atan(2.0 * wheelbase * math.sin(ang) / ld)
        jv = min(j + int(max(0.3 * v, 1.0) / ds), nw - 1)
        acc = min(max(k_v * (vl[jv] - v), a_min), a_max)
        tq = min(max(mass * acc * r0 / n_drv / eP[5], -1.0), 1.0)
        A[t, 0] = min(max(delta / eP[1], -1.0), 1.0)
        off = 1
        if four:
            A[t, 1] = 0.0
            off = 2
        for c in range(off, na):
            A[t, c] = tq
        # --- actuator mapping and model step, as in mppi_numba.rollout
        df_t = A[t, 0] * eP[1]
        dr_t = A[t, 1] * eP[2] if four else 0.0
        T4 = np.zeros(4)
        if layout == 1 or layout == 4:
            T4[2] = A[t, off]
            T4[3] = A[t, off]
        elif layout == 2:
            T4[0] = A[t, off]
            T4[1] = A[t, off]
            T4[2] = A[t, off + 1]
            T4[3] = A[t, off + 1]
        else:
            for i in range(4):
                T4[i] = A[t, off + i]
        sf = sf + min(max(df_t - sf, -eP[3]), eP[3])
        sr = sr + min(max(dr_t - sr, -eP[4]), eP[4])
        for i in range(4):
            ts[i] = ts[i] + min(max(T4[i] * eP[5] - ts[i], -eP[6]), eP[6])
            om = s[3 + i]
            if ts[i] > 0.0:
                f = min(max((eP[7] - om) / eP[8], 0.0), 1.0)
            else:
                f = min(max(om / eP[8], 0.0), 1.0)
            u[4 + i] = ts[i] * f
        u[0] = sf
        u[1] = sf
        u[2] = sr
        u[3] = sr
        s_n, h = _step(s, h, u, mu, n_sub, dt, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ,
                       fzg, kpk_t, tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt)
        vxm = 0.5 * (s[0] + s_n[0])
        vym = 0.5 * (s[1] + s_n[1])
        rm = 0.5 * (s[2] + s_n[2])
        yaw_m = yaw + 0.5 * dt * rm
        x = x + dt * (vxm * math.cos(yaw_m) - vym * math.sin(yaw_m))
        y = y + dt * (vxm * math.sin(yaw_m) + vym * math.cos(yaw_m))
        yaw = yaw + dt * rm
        s = s_n
        best = 1e30
        for w in range(max(0, j - 20), min(nw, j + 80)):
            d2 = (x - txy[w, 0]) ** 2 + (y - txy[w, 1]) ** 2
            if d2 < best:
                best = d2
                j = w
    return A


class MPPIRace2(MPPINumba):
    def __init__(self, env_cfg, K_b=64, K_n=64, sig_bias=(0.3, 0.3, 0.3), bias_white=0.5, ld_min=8.0, ld_t=0.8,
                 k_v=2.0, a_min=-8.0, a_max=4.0, **kw):
        super().__init__(env_cfg, **kw)
        if K_b + K_n >= self.K:
            raise ValueError("K_b + K_n must leave warm-start samples")
        self.K_b, self.K_n, self.bias_white = int(K_b), int(K_n), float(bias_white)
        four = self.layout.startswith("4WS")
        sb = [sig_bias[0]] + ([sig_bias[1]] if four else []) + [sig_bias[2]] * (self.na - (2 if four else 1))
        self.sig_b = np.array(sb, dtype=np.float64)
        self.nom = (float(ld_min), float(ld_t), float(k_v), float(a_min), float(a_max))
        self.r0 = float(self.ph.r0)

    def _window(self, env, mu_hat):
        """Exactly the arrays mppi_numba.MPPINumba.rollout_G passes to the rollout."""
        c, T, tr = self.c, self.T, env.track
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
        return (np.ascontiguousarray(tr.xy[idx], dtype=np.float64), np.ascontiguousarray(tr.psi[idx], dtype=np.float64),
                np.ascontiguousarray(sw), np.ascontiguousarray(bref), np.ascontiguousarray(vl), s0, tr.ds)

    def act(self, env) -> np.ndarray:
        K, T, na = self.K, self.T, self.na
        self.observe(env)
        mu_hat = float(env.mu_hat if env.mu_hat is not None else 0.65)
        eps = self.rng.standard_normal((K, T, na)) * self.sig
        eps[0] = 0.0
        base = np.broadcast_to(self.Un, (K, T, na)).copy()
        Kw = K - self.K_b - self.K_n
        txy, tpsi, sw, bref, vl, s0, ds = self._window(env, mu_hat)
        st0 = np.asarray(env.steer_state, dtype=np.float64)
        ts0 = np.asarray(env.torque_state, dtype=np.float64)
        pose0 = np.array([env.state["x"], env.state["y"], env.state["yaw"]], dtype=np.float64)
        if self.K_b:
            eps[Kw:Kw + self.K_b] = (eps[Kw:Kw + self.K_b] * self.bias_white
                                     + self.rng.standard_normal((self.K_b, 1, na)) * self.sig_b)
        if self.K_n:
            A_nom = nominal_plan(T, na, self.sn_prev, self.hn, st0, ts0, pose0, mu_hat, self.ph.n_sub, self.EP,
                                 *self.MP, txy, tpsi, vl, ds, L_WB, M, self.r0, *self.nom)
            base[Kw + self.K_b:] = A_nom
            eps[Kw + self.K_b] = 0.0
        A = np.clip(base + eps, -1.0, 1.0)
        G_, alive = rollout(np.ascontiguousarray(A), self.sn_prev, self.hn, st0, ts0, pose0,
                            np.asarray(env.prev_a, dtype=np.float64), mu_hat, self.ph.n_sub, self.EP, *self.MP,
                            txy, tpsi, sw, bref, vl, s0)
        w = np.exp((G_ - G_.max()) / self.lam)
        w /= w.sum()
        U = np.tensordot(w, A, axes=(0, 0))
        a0 = U[0].copy()
        self.Un = np.concatenate([U[1:], U[-1:]], 0)
        grp = np.zeros(3)
        grp[0] = w[:Kw].sum()
        grp[1] = w[Kw:Kw + self.K_b].sum()
        grp[2] = w[Kw + self.K_b:].sum()
        self.last = dict(G_best=float(G_.max()), G_mean=float(G_.mean()), alive=float(alive.mean()),
                         ess=float(1.0 / (w ** 2).sum()), w_warm=float(grp[0]), w_bias=float(grp[1]),
                         w_nom=float(grp[2]))
        return a0


class _StubEnv:
    """Minimal stand-in for DDEVEnv (no CarSim): the attributes the planners read."""

    def __init__(self, track, idx, na, dy=0.3, v=20.0, mu_hat=0.8):
        from types import SimpleNamespace
        p = track.xy[idx]
        n = np.array([-math.sin(track.psi[idx]), math.cos(track.psi[idx])])
        self.track, self.idx, self.mu_hat = track, idx, mu_hat
        self.state = dict(x=float(p[0] + dy * n[0]), y=float(p[1] + dy * n[1]), yaw=float(track.psi[idx]), vx=v,
                          vy=0.0, r=0.0, omega=np.full(4, v / 0.29), my_drive=np.zeros(4), ax=0.0, ay=0.0)
        self.cmd = SimpleNamespace(steer=np.zeros(4), motor=np.zeros(4))
        self.steer_state, self.torque_state, self.prev_a = np.zeros(2), np.zeros(4), np.zeros(na)


def check_v1_equivalence(steps=6, layout="4WS_TV"):
    """v2 with K_b = K_n = 0 returns MPPINumba's actions bit for bit; the nominal steers into a left-hander."""
    from trfc_drift.env import EnvConfig
    from trfc_drift.race_tools import race_named
    cfg = EnvConfig(task="race", layout=layout, friction_info="estimate")
    tr = race_named("race_A", w=cfg.width)
    kw = dict(K=64, T=40, n_sub=5, lam=1.0, w_term=2.0, w_lat_term=0.5, v_margin=0.9, preview_m=150.0, w_spin=20.0,
              beta_safe_deg=40.0, beta_everywhere=True, seed=7)
    a, b = MPPINumba(cfg, **kw), MPPIRace2(cfg, K_b=0, K_n=0, **kw)
    env = _StubEnv(tr, 40, a.na)
    diff = 0.0
    for _ in range(steps):
        ua, ub = a.act(env), b.act(env)
        diff = max(diff, float(np.max(np.abs(ua - ub))))
    j_left = int(np.argmax(tr.kappa > 0.02))                      # inside the first left-hander of race_A
    c = MPPIRace2(cfg, K_b=16, K_n=16, **kw)
    env2 = _StubEnv(tr, max(j_left - 30, 0), c.na, dy=0.0, v=15.0)
    c.observe(env2)
    txy, tpsi, sw, bref, vl, s0, ds = c._window(env2, 0.8)
    A_nom = nominal_plan(c.T, c.na, c.sn_prev, c.hn, np.zeros(2), np.zeros(4),
                         np.array([env2.state["x"], env2.state["y"], env2.state["yaw"]]), 0.8, c.ph.n_sub, c.EP,
                         *c.MP, txy, tpsi, vl, ds, L_WB, M, c.r0, *c.nom)
    steer_deg = np.degrees(A_nom[:, 0] * c.EP[1])
    c.act(env2)
    return dict(max_abs_action_diff_v1_v2off=diff, nominal_steer_deg_mean=float(steer_deg.mean()),
                nominal_steer_deg_max=float(steer_deg.max()), v2_last=c.last)


if __name__ == "__main__":
    import json
    print(json.dumps(check_v1_equivalence(), indent=1))
