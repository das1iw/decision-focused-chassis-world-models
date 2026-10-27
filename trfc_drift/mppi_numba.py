"""Compiled (numba) MPPI rollout for the physics-v2 world model: the same planner as mppi.MPPI, 10-30x faster on CPU.

Scalar re-implementation of:
- wm_physics2.DDEVPhysics2.substep: loads, compliance steer, kinematics, lagged slips, aero, motor, wheel spin;
- tire_bakker.BakkerTire.combined in mode "scaled", with the aligning moment;
- the MPPI rollout of mppi.MPPI / mppi_gpu.MPPIGraph: action mapping, pose kinematics, track projection, reward,
  sideslip barrier, friction-envelope and lateral terminal costs.
It runs in float64 with prange over samples, and must agree with the torch reference: check_numba_parity() compares
predicted states and returns on random inputs (run it after any change to either implementation).

Everything the kernel needs is packed by pack_model(physics2) and pack_env(cfg) into plain arrays.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
import torch
from numba import njit, prange

from trfc_drift.mppi import MPPI, envelope_vlim
from trfc_drift.wm_physics2 import DDEVPhysics2, IZZ, L_WB, M, X_AERO

G_ACC = 9.80665
LAYOUT_CODE = {"4WS_TV": 0, "2WS_RWD": 1, "2WS_AWD": 2, "2WS_TV": 3, "4WS_RWD": 4}


# ------------------------------------------------------------------------------------------------ table helpers
@njit(cache=True, inline="always")
def _search(xs, x):
    n = xs.shape[0]
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    i = lo - 1
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    return i


@njit(cache=True, inline="always")
def _lerp(xs, ys, x):
    x = min(max(x, xs[0]), xs[-1])
    i = _search(xs, x)
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])


@njit(cache=True, inline="always")
def _bil(rows, cols, tab, r, c):
    r = min(max(r, rows[0]), rows[-1])
    c = min(max(c, cols[0]), cols[-1])
    i = _search(rows, r)
    j = _search(cols, c)
    tr = (r - rows[i]) / (rows[i + 1] - rows[i])
    tc = (c - cols[j]) / (cols[j + 1] - cols[j])
    return ((1 - tr) * (1 - tc) * tab[i, j] + tr * (1 - tc) * tab[i + 1, j]
            + (1 - tr) * tc * tab[i, j + 1] + tr * tc * tab[i + 1, j + 1])


@njit(cache=True, inline="always")
def _sign(x):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


# ------------------------------------------------------------------------------------------------ tire
@njit(cache=True)
def _tire(kappa, alpha, fz, mu, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk_t, tpk_t, q1):
    eps = 1e-9
    rx = mu
    ry = mu
    kappa = min(max(kappa, -0.999), 50.0)
    kpk = _lerp(fzg, kpk_t, fz)
    tpk = _lerp(fzg, tpk_t, fz)
    drive = kappa >= 0.0
    sx = abs(kappa) / (1.0 + kappa)
    sy = abs(math.tan(alpha)) / (1.0 + kappa)
    if drive:
        sxm = rx * kpk / (1.0 + rx * kpk)
    else:
        sxm = rx * kpk / (1.0 - rx * kpk)
    sym = ry * tpk
    nx = sx / sxm
    ny = sy / sym
    st = math.sqrt(nx * nx + ny * ny) + eps
    a = st * sxm
    if drive:
        kq = a / max(1.0 - a, 1e-3)
    else:
        kq = a / (1.0 + a)
    aq = math.atan(st * sym)
    fx0 = _bil(kx, fzx, FX, kq / rx, fz)
    fy0 = _bil(ay, fzy, FY, aq / ry, fz)
    e = min(st, 1.0)
    fx0s = fx0 - e * (fx0 - fy0) * (ny / st) ** 2
    fy0s = fy0 - e * (fy0 - fx0) * (nx / st) ** 2
    eta = math.atan2(ny, nx)
    theta = math.atan2(sy, sx)
    lam = eta + 2.0 * (theta - eta) / math.pi * math.atan(q1 * st * st)
    fx = _sign(kappa) * fx0s * rx * math.cos(lam)
    fy = -_sign(alpha) * fy0s * ry * math.sin(lam)
    mz = _sign(alpha) * ry * _bil(am, fzm, MZ, aq / ry, fz) * math.sin(lam)
    return fx, fy, mz


# ------------------------------------------------------------------------------------------------ physics substep
@njit(cache=True)
def _substep(s, h, u, mu, dt, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk_t, tpk_t,
             m_rad, m_tq, ag, cxt, cyt, cmzt, s_out, h_out):
    """sP: [R, jw, r0, kz, rr_c, rr_v, tc_m, rho_a, vlow, kappa_dec, tc_load, use_mz, use_comp, q1]"""
    R, jw, r0, kz, rr_c, rr_v, tc_m, rho_a, vlow = sP[0], sP[1], sP[2], sP[3], sP[4], sP[5], sP[6], sP[7], sP[8]
    kappa_dec, tc_load, use_mz, use_comp, q1 = sP[9], sP[10], sP[11], sP[12], sP[13]
    vx, vy, r = s[0], s[1], s[2]
    axf, ayf = h[8], h[9]
    fxs = 0.0
    fys = 0.0
    mzs = 0.0
    for i in range(4):
        fz = load[i, 0] + load[i, 1] * axf + load[i, 2] * ayf
        if fz < 50.0:
            fz = 50.0
        stw = u[i]
        if use_comp > 0.5:
            stw += (steer_c[i, 0] + steer_c[i, 1] * h[14 + i] + steer_c[i, 2] * h[10 + i] + steer_c[i, 3] * fz
                    + steer_c[i, 4] * ayf) * (math.pi / 180.0)
        vix = vx - r * yw[i]
        viy = vy + r * xw[i]
        cd = math.cos(stw)
        sd = math.sin(stw)
        vl = vix * cd + viy * sd
        vt = -vix * sd + viy * cd
        va = max(abs(vl), vlow)
        om = s[3 + i]
        kap = (om * R - vl) / va
        tau = h[i]
        kl = h[4 + i]
        fx, fy, mz = _tire(kl, math.atan(tau), fz, mu, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk_t, tpk_t, q1)
        if use_mz < 0.5:
            mz = 0.0
        fxb = fx * cd - fy * sd
        fyb = fx * sd + fy * cd
        fxs += fxb
        fys += fyb
        mzs += xw[i] * fyb - yw[i] * fxb + mz
        # motor, wheel spin
        tq = s[7 + i]
        tmax = _lerp(m_rad, m_tq, abs(om))
        tc = min(max(u[4 + i], -tmax), tmax)
        arm = r0 - fz / kz
        myrr = fz * (rr_c + rr_v * va * 3.6) * math.tanh(om / 1.0)
        s_out[3 + i] = om + dt * (tq - fx * arm - myrr) / jw
        s_out[7 + i] = tq + dt * (tc - tq) / tc_m
        # slip lags
        dec = (abs(kap) < abs(kl)) and (_sign(kap) == _sign(kl))
        g = math.exp(-va * dt)
        if dec:
            if kappa_dec > 0.0:
                g = math.exp(-va * dt / kappa_dec)
            else:
                g = 0.0
        h_out[4 + i] = g * kl + (1.0 - g) * kap
        ga = math.exp(-va * dt)
        h_out[i] = ga * tau + (1.0 - ga) * (vt / va)
        h_out[10 + i] = fx
        h_out[14 + i] = fy
    # aero (aero slip = -beta)
    beta_deg = math.degrees(math.atan2(vy, max(abs(vx), 0.5)))
    q = rho_a * (vx * vx + vy * vy)
    cx = _lerp(ag, cxt, -beta_deg)
    cy = _lerp(ag, cyt, -beta_deg)
    cmz = _lerp(ag, cmzt, -beta_deg)
    fxa = -cx * q * _sign(vx)
    fya = cy * q
    mza = cmz * q * L_WB + X_AERO * fya
    axx = (fxs + fxa) / M
    ayy = (fys + fya) / M
    s_out[0] = vx + dt * (axx + vy * r)
    s_out[1] = vy + dt * (ayy - vx * r)
    s_out[2] = r + dt * (mzs + mza) / IZZ
    gl = math.exp(-dt / tc_load)
    h_out[8] = gl * axf + (1.0 - gl) * axx
    h_out[9] = gl * ayf + (1.0 - gl) * ayy


@njit(cache=True)
def _step(s, h, u, mu, n_sub, dt, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk_t,
          tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt):
    hs = dt / n_sub
    s1 = s.copy()
    h1 = h.copy()
    s2 = np.empty(11)
    h2 = np.empty(18)
    for _ in range(n_sub):
        _substep(s1, h1, u, mu, hs, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk_t,
                 tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt, s2, h2)
        s1[:] = s2
        h1[:] = h2
    return s1, h1


# ------------------------------------------------------------------------------------------------ MPPI rollout
@njit(cache=True, parallel=True)
def rollout(A, s0, h0, st0, ts0, pose0, aprev0, mu, n_sub, eP, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY,
            am, fzm, MZ, fzg, kpk_t, tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt, txy, tpsi, tsw, bref, vl, s_prev0):
    """A (K, T, na) normalized actions -> returns (G, alive). eP:
    [dt, steer_f_max, steer_r_max, rate_f, rate_r, torque_max, torque_step, omega_lim, omega_band, layout,
     off_limit, drift, beta_everywhere, w_spin, beta_safe, w_term, w_lat_term]"""
    K, T, na = A.shape
    dt = eP[0]
    layout = int(eP[9])
    four = layout == 0 or layout == 4
    G = np.zeros(K)
    alive_out = np.ones(K)
    nw = txy.shape[0]
    for k in prange(K):
        s = s0.copy()
        h = h0.copy()
        sf, sr = st0[0], st0[1]
        ts = ts0.copy()
        x, y, yaw = pose0[0], pose0[1], pose0[2]
        s_prev = s_prev0
        u = np.empty(8)
        g_acc = 0.0
        alive = True
        e_lat = 0.0
        e_psi = 0.0
        beta = 0.0
        j = 0
        for t in range(T):
            # --- action mapping (DDEVEnv._apply)
            a = np.empty(na)
            for c in range(na):
                a[c] = min(max(A[k, t, c], -1.0), 1.0)
            df_t = a[0] * eP[1]
            dr_t = a[1] * eP[2] if four else 0.0
            off = 2 if four else 1
            T4 = np.zeros(4)
            if layout == 1 or layout == 4:          # RWD
                T4[2] = a[off]
                T4[3] = a[off]
            elif layout == 2:                        # AWD
                T4[0] = a[off]
                T4[1] = a[off]
                T4[2] = a[off + 1]
                T4[3] = a[off + 1]
            else:                                    # TV
                for i in range(4):
                    T4[i] = a[off + i]
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
            # --- world model step
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
            # --- track projection (nearest window sample)
            best = 1e30
            for w in range(nw):
                d2 = (x - txy[w, 0]) ** 2 + (y - txy[w, 1]) ** 2
                if d2 < best:
                    best = d2
                    j = w
            tx = math.cos(tpsi[j])
            ty = math.sin(tpsi[j])
            dx = x - txy[j, 0]
            dy = y - txy[j, 1]
            e_lat = -dx * ty + dy * tx
            s_tr = tsw[j] + dx * tx + dy * ty
            e_psi = math.atan2(math.sin(yaw - tpsi[j]), math.cos(yaw - tpsi[j]))
            beta = math.atan2(s[1], max(s[0], 0.5))
            p = (s_tr - s_prev) / (dt * 20.0)
            s_prev = s_tr
            q = math.exp(-(e_lat / 2.0) ** 2)
            if eP[11] > 0.5:
                br = bref[j]
                if eP[12] > 0.5 or abs(br) > 1e-6:
                    q *= math.exp(-((beta - br) / math.radians(15.0)) ** 2)
            da2 = 0.0
            for c in range(na):
                prev = aprev0[c] if t == 0 else A[k, t - 1, c]
                d = A[k, t, c] - prev
                da2 += d * d
            rew = p * q - 0.01 * da2
            if eP[13] > 0.0:
                ex = abs(beta) - eP[14]
                if ex > 0.0:
                    rew -= eP[13] * ex * ex
            finite = True
            for i in range(11):
                if not math.isfinite(s[i]):
                    finite = False
            dead = (abs(e_lat) > eP[10]) or (abs(beta) > math.radians(70.0)) or \
                   (abs(e_psi) > math.radians(110.0)) or (not finite)
            if alive:
                if dead:
                    rew -= 10.0
                g_acc += rew
            if dead:
                alive = False
        if alive:
            if eP[15] > 0.0:
                speed = math.sqrt(s[0] * s[0] + s[1] * s[1])
                ex = speed - vl[j]
                if ex > 0.0:
                    g_acc -= eP[15] * ex * ex
            if eP[16] > 0.0:
                g_acc -= eP[16] * (e_lat * e_lat + (5.0 * math.sin(e_psi + beta)) ** 2)
        if not math.isfinite(g_acc):
            g_acc = -1e6
        G[k] = g_acc
        alive_out[k] = 1.0 if alive else 0.0
    return G, alive_out


# ------------------------------------------------------------------------------------------------ packing
def pack_model(ph: DDEVPhysics2):
    f = lambda t: np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float64))  # noqa: E731
    tb = ph.tire
    sP = np.array([ph.R, ph.jw, ph.r0, ph.kz, ph.rr_c, ph.rr_v, ph.tc_m, ph.rho_a, ph.vlow, ph.kappa_dec, ph.tc_load,
                   1.0 if ph.use_mz else 0.0, 1.0 if ph.use_comp else 0.0, tb.q1], dtype=np.float64)
    return (sP, f(ph.load), f(ph.steer_c), f(ph.x), f(ph.y), f(tb.kx), f(tb.fzx), f(tb.FX), f(tb.ay), f(tb.fzy),
            f(tb.FY), f(tb.am), f(tb.fzm), f(tb.MZ), f(tb.fzg), f(tb.kpk), f(tb.tan_apk), f(ph.m_rad), f(ph.m_tq),
            f(ph.ag), f(ph.cx), f(ph.cy), f(ph.cmz))


def pack_env(c, beta_everywhere, w_spin, beta_safe_deg, w_term, w_lat_term):
    return np.array([c.dt, math.radians(c.steer_f_max_deg), math.radians(c.steer_r_max_deg),
                     math.radians(c.steer_rate_f_dps) * c.dt, math.radians(c.steer_rate_r_dps) * c.dt, c.torque_max,
                     c.torque_rate * c.dt, c.omega_lim, c.omega_band, LAYOUT_CODE[c.layout], c.width + c.offtrack_margin,
                     1.0 if (c.task == "drift" or beta_everywhere) else 0.0, 1.0 if beta_everywhere else 0.0, w_spin,
                     math.radians(beta_safe_deg), w_term, w_lat_term], dtype=np.float64)


class MPPINumba(MPPI):
    """mppi.MPPI with the sample rollout and the per-step hidden-state update compiled by numba (float64)."""

    def __init__(self, env_cfg, span=None, **kw):
        kw["device"] = "cpu"
        kw["dtype"] = torch.float64
        super().__init__(env_cfg, **kw)
        self.MP = pack_model(self.ph)
        self.EP = pack_env(env_cfg, self.beta_everywhere, self.w_spin, self.beta_safe_deg, self.w_term,
                           self.w_lat_term)
        self.rng = np.random.default_rng(kw.get("seed", 0))
        self.sig = self.sigma.cpu().numpy().astype(np.float64)
        self.Un = np.zeros((self.T, self.na))
        self.span = span or int((40.0 * self.T * env_cfg.dt + self.preview_m + 20.0) / 0.5)
        self.hn = None
        self.sn_prev = None

    def reset(self):
        super().reset()
        self.Un[:] = 0.0
        self.hn = None
        self.sn_prev = None

    def _state_np(self, env):
        st = env.state
        return np.concatenate([[st["vx"], st["vy"], st["r"]], st["omega"], st["my_drive"]]).astype(np.float64)

    def observe(self, env):
        s = self._state_np(env)
        mu = float(env.mu_hat if env.mu_hat is not None else 0.65)
        u = np.concatenate([env.cmd.steer, env.cmd.motor]).astype(np.float64)
        if self.hn is None:
            h = self.ph.init_hidden(torch.as_tensor(s)[None], torch.as_tensor(u)[None],
                                    torch.tensor([mu], dtype=torch.float64),
                                    torch.tensor([env.state["ax"]], dtype=torch.float64),
                                    torch.tensor([env.state["ay"]], dtype=torch.float64))
            self.hn = h[0].numpy().astype(np.float64)
        else:
            _, self.hn = _step(self.sn_prev, self.hn, u, mu, self.ph.n_sub, self.c.dt, *self.MP)
        self.sn_prev = s

    def rollout_G(self, env, A, mu_hat):
        """(G, alive) for action sequences A (K, T, na), float64, from the current observed state."""
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
        return rollout(np.ascontiguousarray(A, dtype=np.float64), self.sn_prev, self.hn,
                       np.asarray(env.steer_state, dtype=np.float64), np.asarray(env.torque_state, dtype=np.float64),
                       np.array([env.state["x"], env.state["y"], env.state["yaw"]], dtype=np.float64),
                       np.asarray(env.prev_a, dtype=np.float64), mu_hat, self.ph.n_sub, self.EP, *self.MP,
                       np.ascontiguousarray(tr.xy[idx], dtype=np.float64),
                       np.ascontiguousarray(tr.psi[idx], dtype=np.float64), np.ascontiguousarray(sw),
                       np.ascontiguousarray(bref), np.ascontiguousarray(vl), s0)

    def act(self, env) -> np.ndarray:
        K, T = self.K, self.T
        self.observe(env)
        mu_hat = float(env.mu_hat if env.mu_hat is not None else 0.65)
        eps = self.rng.standard_normal((K, T, self.na)) * self.sig
        eps[0] = 0.0
        A = np.clip(self.Un[None] + eps, -1.0, 1.0)
        G_, alive = self.rollout_G(env, A, mu_hat)
        w = np.exp((G_ - G_.max()) / self.lam)
        w /= w.sum()
        U = np.tensordot(w, A, axes=(0, 0))
        a0 = U[0].copy()
        self.Un = np.concatenate([U[1:], U[-1:]], 0)
        self.last = dict(G_best=float(G_.max()), G_mean=float(G_.mean()), alive=float(alive.mean()),
                         ess=float(1.0 / (w ** 2).sum()))
        return a0


def check_numba_parity(K=64, T=40, seed=0):
    """Torch reference (mppi.MPPI, float64) vs numba on random inputs: hidden-state update and returns G."""
    import types
    from trfc_drift import env as E
    from trfc_drift.evaluate import get_track
    cfg = E.EnvConfig(task="drift", layout="4WS_TV", friction_info="estimate", probe=True)
    tr = get_track("dev_9001", cfg.width).anchored(0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    kw = dict(K=K, T=T, n_sub=5, lam=1.0, w_term=2.0, w_lat_term=0.5, v_margin=0.9, w_spin=20.0,
              beta_everywhere=True, seed=seed)
    ref = MPPI(cfg, device="cpu", dtype=torch.float64, **kw)
    nb_ = MPPINumba(cfg, **kw)
    out = {}
    st = dict(vx=17.0, vy=1.5, r=-0.3, omega=np.array([58.0, 60.0, 70.0, 75.0]), my_drive=np.array([100., 150., 400., 450.]),
              ax=-1.0, ay=3.0, x=20.0, y=0.4, yaw=0.05)
    env = types.SimpleNamespace(state=st, cmd=types.SimpleNamespace(steer=np.array([-0.05, -0.05, 0.02, 0.02]),
                                                                    motor=np.array([100., 150., 400., 450.])),
                                mu_hat=0.55, steer_state=np.array([-0.05, 0.02]),
                                torque_state=np.array([100., 150., 400., 450.]), prev_a=rng.uniform(-1, 1, 6),
                                track=tr, idx=40)
    for step in range(4):          # exercise the teacher-forced hidden update
        ref.observe(env)
        nb_.observe(env)
        env.state = dict(env.state, vx=env.state["vx"] - 0.05, r=env.state["r"] - 0.01,
                         omega=env.state["omega"] + rng.normal(0, 1.0, 4))
        env.cmd.motor = env.cmd.motor + rng.normal(0, 50.0, 4)
    ref.observe(env)
    nb_.observe(env)
    out["hidden_max_abs"] = float(np.max(np.abs(ref.h[0].numpy() - nb_.hn)))
    A = np.clip(rng.normal(0, 0.5, (K, T, 6)), -1, 1)
    G_ref, alive_ref = ref._rollout(env, torch.as_tensor(A), 0.55)
    G_nb, alive_nb = nb_.rollout_G(env, A, 0.55)
    out["G_max_abs"] = float(np.max(np.abs(G_ref.numpy() - G_nb)))
    out["G_scale"] = float(np.max(np.abs(G_ref.numpy())))
    out["alive_mismatch"] = int(np.sum(alive_ref.numpy().astype(float) != alive_nb))
    out["alive_frac"] = float(alive_nb.mean())
    return out
