"""GPU (numba-CUDA) port of the physics-v2 MPPI rollout of the drift/race planner (research/284, Track B).

A line-by-line transcription of trfc_drift/mppi_numba.py (_search, _lerp, _bil, _sign, _tire, _substep, _step and
rollout), with numpy temporaries replaced by per-thread local arrays and one CUDA thread per (episode, sample).
Several episodes are planned at once: per-episode initial states, hidden states, actuator states, poses, previous
actions, friction and track windows. float64 throughout, so the kernel can be checked against the CPU numba rollout
on identical inputs (tools/drift284_check_cuda_v1.py). mppi_numba.py itself is unchanged.
"""
from __future__ import annotations

import math

import numpy as np
from numba import cuda, float64

from trfc_drift.wm_physics2 import IZZ, L_WB, M, X_AERO


@cuda.jit(device=True)
def _mx(a, b):
    """Python's max(a, b) (numba-CUDA 0.67 mis-types the builtin in device functions here)."""
    return b if b > a else a


@cuda.jit(device=True)
def _mn(a, b):
    """Python's min(a, b)."""
    return b if b < a else a


@cuda.jit(device=True)
def _search(xs, x):
    n = xs.shape[0]
    lo = 0
    hi = n
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


@cuda.jit(device=True)
def _lerp(xs, ys, x):
    x = _mn(_mx(x, xs[0]), xs[xs.shape[0] - 1])
    i = _search(xs, x)
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])


@cuda.jit(device=True)
def _bil(rows, cols, tab, r, c):
    r = _mn(_mx(r, rows[0]), rows[rows.shape[0] - 1])
    c = _mn(_mx(c, cols[0]), cols[cols.shape[0] - 1])
    i = _search(rows, r)
    j = _search(cols, c)
    tr = (r - rows[i]) / (rows[i + 1] - rows[i])
    tc = (c - cols[j]) / (cols[j + 1] - cols[j])
    return ((1 - tr) * (1 - tc) * tab[i, j] + tr * (1 - tc) * tab[i + 1, j]
            + (1 - tr) * tc * tab[i, j + 1] + tr * tc * tab[i + 1, j + 1])


@cuda.jit(device=True)
def _sign(x):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@cuda.jit(device=True)
def _tire(kappa, alpha, fz, mu, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk_t, tpk_t, q1):
    eps = 1e-9
    rx = mu
    ry = mu
    kappa = _mn(_mx(kappa, -0.999), 50.0)
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
        kq = a / _mx(1.0 - a, 1e-3)
    else:
        kq = a / (1.0 + a)
    aq = math.atan(st * sym)
    fx0 = _bil(kx, fzx, FX, kq / rx, fz)
    fy0 = _bil(ay, fzy, FY, aq / ry, fz)
    e = _mn(st, 1.0)
    fx0s = fx0 - e * (fx0 - fy0) * (ny / st) ** 2
    fy0s = fy0 - e * (fy0 - fx0) * (nx / st) ** 2
    eta = math.atan2(ny, nx)
    theta = math.atan2(sy, sx)
    lam = eta + 2.0 * (theta - eta) / math.pi * math.atan(q1 * st * st)
    fx = _sign(kappa) * fx0s * rx * math.cos(lam)
    fy = -_sign(alpha) * fy0s * ry * math.sin(lam)
    mz = _sign(alpha) * ry * _bil(am, fzm, MZ, aq / ry, fz) * math.sin(lam)
    return fx, fy, mz


@cuda.jit(device=True)
def _substep(s, h, u, mu, dt, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk_t, tpk_t,
             m_rad, m_tq, ag, cxt, cyt, cmzt, s_out, h_out):
    R = sP[0]; jw = sP[1]; r0 = sP[2]; kz = sP[3]; rr_c = sP[4]; rr_v = sP[5]; tc_m = sP[6]; rho_a = sP[7]
    vlow = sP[8]; kappa_dec = sP[9]; tc_load = sP[10]; use_mz = sP[11]; use_comp = sP[12]; q1 = sP[13]
    vx = s[0]
    vy = s[1]
    r = s[2]
    axf = h[8]
    ayf = h[9]
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
        va = _mx(abs(vl), vlow)
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
        tq = s[7 + i]
        tmax = _lerp(m_rad, m_tq, abs(om))
        tc = _mn(_mx(u[4 + i], -tmax), tmax)
        arm = r0 - fz / kz
        myrr = fz * (rr_c + rr_v * va * 3.6) * math.tanh(om / 1.0)
        s_out[3 + i] = om + dt * (tq - fx * arm - myrr) / jw
        s_out[7 + i] = tq + dt * (tc - tq) / tc_m
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
    beta_deg = math.degrees(math.atan2(vy, _mx(abs(vx), 0.5)))
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


@cuda.jit(device=True)
def _step_inplace(s, h, u, mu, n_sub, dt, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg,
                  kpk_t, tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt, s2, h2):
    hs = dt / n_sub
    for _ in range(n_sub):
        _substep(s, h, u, mu, hs, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg, kpk_t,
                 tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt, s2, h2)
        for i in range(11):
            s[i] = s2[i]
        for i in range(18):
            h[i] = h2[i]


@cuda.jit
def rollout_kernel(A, s0, h0, st0, ts0, pose0, aprev0, mu, n_sub, eP, sP, load, steer_c, xw, yw, kx, fzx, FX, ay,
                   fzy, FY, am, fzm, MZ, fzg, kpk_t, tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt, txy, tpsi, tsw, bref, vl,
                   nwin, s_prev0, G, alive_out):
    """A (E, K, T, na); per-episode s0 (E,11), h0 (E,18), st0 (E,2), ts0 (E,4), pose0 (E,3), aprev0 (E,na), mu (E),
    track windows txy (E,NW,2), tpsi/tsw/bref/vl (E,NW), nwin (E) valid lengths, s_prev0 (E). Out G, alive (E,K)."""
    E = A.shape[0]
    K = A.shape[1]
    T = A.shape[2]
    na = A.shape[3]
    tid = cuda.grid(1)
    if tid >= E * K:
        return
    e = tid // K
    k = tid % K
    s = cuda.local.array(11, float64)
    h = cuda.local.array(18, float64)
    s2 = cuda.local.array(11, float64)
    h2 = cuda.local.array(18, float64)
    u = cuda.local.array(8, float64)
    a = cuda.local.array(8, float64)
    T4 = cuda.local.array(4, float64)
    ts = cuda.local.array(4, float64)
    for i in range(11):
        s[i] = s0[e, i]
    for i in range(18):
        h[i] = h0[e, i]
    dt = eP[0]
    layout = int(eP[9])
    four = layout == 0 or layout == 4
    sf = st0[e, 0]
    sr = st0[e, 1]
    for i in range(4):
        ts[i] = ts0[e, i]
    x = pose0[e, 0]
    y = pose0[e, 1]
    yaw = pose0[e, 2]
    s_prev = s_prev0[e]
    mue = mu[e]
    nw = nwin[e]
    g_acc = 0.0
    alive = True
    e_lat = 0.0
    e_psi = 0.0
    beta = 0.0
    j = 0
    for t in range(T):
        for c in range(na):
            a[c] = _mn(_mx(A[e, k, t, c], -1.0), 1.0)
        df_t = a[0] * eP[1]
        dr_t = a[1] * eP[2] if four else 0.0
        off = 2 if four else 1
        for i in range(4):
            T4[i] = 0.0
        if layout == 1 or layout == 4:
            T4[2] = a[off]
            T4[3] = a[off]
        elif layout == 2:
            T4[0] = a[off]
            T4[1] = a[off]
            T4[2] = a[off + 1]
            T4[3] = a[off + 1]
        else:
            for i in range(4):
                T4[i] = a[off + i]
        sf = sf + _mn(_mx(df_t - sf, -eP[3]), eP[3])
        sr = sr + _mn(_mx(dr_t - sr, -eP[4]), eP[4])
        for i in range(4):
            ts[i] = ts[i] + _mn(_mx(T4[i] * eP[5] - ts[i], -eP[6]), eP[6])
            om = s[3 + i]
            if ts[i] > 0.0:
                f = _mn(_mx((eP[7] - om) / eP[8], 0.0), 1.0)
            else:
                f = _mn(_mx(om / eP[8], 0.0), 1.0)
            u[4 + i] = ts[i] * f
        u[0] = sf
        u[1] = sf
        u[2] = sr
        u[3] = sr
        vx0 = s[0]
        vy0 = s[1]
        r0_ = s[2]
        _step_inplace(s, h, u, mue, n_sub, dt, sP, load, steer_c, xw, yw, kx, fzx, FX, ay, fzy, FY, am, fzm, MZ, fzg,
                      kpk_t, tpk_t, m_rad, m_tq, ag, cxt, cyt, cmzt, s2, h2)
        vxm = 0.5 * (vx0 + s[0])
        vym = 0.5 * (vy0 + s[1])
        rm = 0.5 * (r0_ + s[2])
        yaw_m = yaw + 0.5 * dt * rm
        x = x + dt * (vxm * math.cos(yaw_m) - vym * math.sin(yaw_m))
        y = y + dt * (vxm * math.sin(yaw_m) + vym * math.cos(yaw_m))
        yaw = yaw + dt * rm
        best = 1e30
        for w in range(nw):
            d2 = (x - txy[e, w, 0]) ** 2 + (y - txy[e, w, 1]) ** 2
            if d2 < best:
                best = d2
                j = w
        tx = math.cos(tpsi[e, j])
        ty = math.sin(tpsi[e, j])
        dx = x - txy[e, j, 0]
        dy = y - txy[e, j, 1]
        e_lat = -dx * ty + dy * tx
        s_tr = tsw[e, j] + dx * tx + dy * ty
        e_psi = math.atan2(math.sin(yaw - tpsi[e, j]), math.cos(yaw - tpsi[e, j]))
        beta = math.atan2(s[1], _mx(s[0], 0.5))
        p = (s_tr - s_prev) / (dt * 20.0)
        s_prev = s_tr
        q = math.exp(-(e_lat / 2.0) ** 2)
        if eP[11] > 0.5:
            br = bref[e, j]
            if eP[12] > 0.5 or abs(br) > 1e-6:
                q *= math.exp(-((beta - br) / math.radians(15.0)) ** 2)
        da2 = 0.0
        for c in range(na):
            prev = aprev0[e, c] if t == 0 else A[e, k, t - 1, c]
            d = A[e, k, t, c] - prev
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
            ex = speed - vl[e, j]
            if ex > 0.0:
                g_acc -= eP[15] * ex * ex
        if eP[16] > 0.0:
            g_acc -= eP[16] * (e_lat * e_lat + (5.0 * math.sin(e_psi + beta)) ** 2)
    if not math.isfinite(g_acc):
        g_acc = -1e6
    G[e, k] = g_acc
    alive_out[e, k] = 1.0 if alive else 0.0


def rollout_gpu(A, s0, h0, st0, ts0, pose0, aprev0, mu, n_sub, EP, MP, txy, tpsi, tsw, bref, vl, nwin, s_prev0,
                threads=128):
    """Host wrapper: all array arguments may be numpy or numba device arrays; returns device arrays (G, alive)."""
    E, K = A.shape[0], A.shape[1]
    d = lambda x: x if cuda.is_cuda_array(x) else cuda.to_device(np.ascontiguousarray(x))  # noqa: E731
    G = cuda.device_array((E, K), dtype=np.float64)
    alive = cuda.device_array((E, K), dtype=np.float64)
    blocks = (E * K + threads - 1) // threads
    rollout_kernel[blocks, threads](d(A), d(s0), d(h0), d(st0), d(ts0), d(pose0), d(aprev0), d(mu), n_sub, d(EP),
                                    *[d(m) for m in MP], d(txy), d(tpsi), d(tsw), d(bref), d(vl), d(nwin), d(s_prev0),
                                    G, alive)
    return G, alive
