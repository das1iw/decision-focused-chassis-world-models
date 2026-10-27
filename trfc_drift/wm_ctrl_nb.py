"""Numba port of the Simulink physics chassis controller (research/135 M3), for the batched world-model environment.

Same equations and parameters as trfc_drift/cl_controller.py (the Python mirror, itself equal to the Simulink model
DDEV_ClosedLoop_CarSim to about 1e-15). Inputs are the controller state x (21, SI, canonical wheel order; only vx, r,
ax, ay, beta are used) and the host's reference features ref (simulink_host.N_REF). check_parity() compares this port
with the mirror on random inputs.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# parameter vector layout (pack_params)
_KEYS = ("m", "L", "a", "b", "h", "tf", "tr", "Rw", "Iz", "g", "rollShareFront", "Cf", "Cr", "Pmax", "Tmax",
         "steerFMax", "steerRMax", "vMax", "cLat", "cTot", "kV", "kR", "kLd", "LdMin", "LdMax", "KrsMin", "KrsMax",
         "betaMax", "vBetaMin", "rMaxShare")
PI = {k: i for i, k in enumerate(_KEYS)}


def pack_params(P=None) -> tuple[np.ndarray, np.ndarray]:
    """(scalar parameter vector, preview stations) from a cl_controller.Params."""
    from trfc_drift import cl_controller as CC
    P = CC.P0 if P is None else P
    return (np.array([float(getattr(P, k)) for k in _KEYS], dtype=np.float64),
            np.ascontiguousarray(np.asarray(P.preview, dtype=np.float64)))


@njit(cache=True)
def load_estimator(x, p):
    m, L, a, b, h = p[0], p[1], p[2], p[3], p[4]
    tf, tr, g, sh = p[5], p[6], p[9], p[10]
    ax, ay = x[3], x[4]
    Ff = m * (g * b - ax * h) / L
    Fr = m * (g * a + ax * h) / L
    dFf = sh * m * ay * h / tf
    dFr = (1.0 - sh) * m * ay * h / tr
    Fz = np.empty(4)
    Fz[0] = Ff / 2 - dFf
    Fz[1] = Ff / 2 + dFf
    Fz[2] = Fr / 2 - dFr
    Fz[3] = Fr / 2 + dFr
    lo = 0.05 * m * g / 4
    for i in range(4):
        if Fz[i] < lo:
            Fz[i] = lo
    return Fz


@njit(cache=True)
def reference(x, ref, p, s):
    n = s.shape[0]
    m, g, Pmax, vMax = p[0], p[9], p[13], p[17]
    cLat, cTot, kLd, LdMin, LdMax, rMaxShare = p[18], p[19], p[22], p[23], p[24], p[29]
    task = ref[0]
    mu = max(ref[1], 0.2)
    e_lat, e_psi, v_cap, active = ref[2], ref[3], ref[5], ref[6]
    vx = max(x[0], 1.0)
    a_tot = cTot * mu * g
    v = np.empty(n)
    for i in range(n):
        v[i] = min(math.sqrt(cLat * mu * g / max(abs(ref[8 + i]), 1e-4)), vMax)
    for i in range(n - 2, -1, -1):
        a_lat = v[i + 1] ** 2 * abs(ref[8 + i + 1])
        ab = math.sqrt(max(a_tot ** 2 - a_lat ** 2, (0.05 * a_tot) ** 2))
        ab = min(ab, Pmax / (m * max(v[i + 1], 5.0)))
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * ab * (s[i + 1] - s[i])))
    if v_cap > 0:
        for i in range(n):
            v[i] = min(v[i], v_cap)
    vT = v[0]
    aFF = (v[1] ** 2 - v[0] ** 2) / (2.0 * (s[1] - s[0]))
    Ld = min(max(kLd * vx, LdMin), LdMax)
    j = int(min(max(math.floor(Ld / (s[1] - s[0])) + 1, 1), n - 1)) - 1
    f = (Ld - s[j]) / (s[j + 1] - s[j])
    xt = ref[8 + n + j] + f * (ref[8 + n + j + 1] - ref[8 + n + j])
    yt = ref[8 + 2 * n + j] + f * (ref[8 + 2 * n + j + 1] - ref[8 + 2 * n + j])
    kpp = 2.0 * yt / max(xt ** 2 + yt ** 2, 1.0)
    r_max = rMaxShare * mu * g / vx
    r_ref = min(max(vx * kpp, -r_max), r_max)
    out = np.empty(10)
    out[0], out[1], out[2], out[3], out[4] = vT, aFF, kpp, Ld, r_ref
    out[5], out[6], out[7], out[8], out[9] = mu, active, task, e_lat, e_psi
    return out


@njit(cache=True)
def motion_controller(x, r, p):
    m, L, a, b, Iz, g = p[0], p[1], p[2], p[3], p[8], p[9]
    Cf, Cr, Pmax = p[11], p[12], p[13]
    sFM, sRM, cTot, kV, kR, Kmin, Kmax = p[15], p[16], p[19], p[20], p[21], p[25], p[26]
    vx, yaw_rate, ay = max(x[0], 1.0), x[2], x[4]
    vT, aFF, kpp, r_ref, mu = r[0], r[1], r[2], r[4], r[5]
    a_tot = cTot * mu * g
    a_avail = math.sqrt(max(a_tot ** 2 - ay ** 2, (0.1 * a_tot) ** 2))
    a_pow = Pmax / (m * max(vx, 5.0))
    aD = min(a_avail, a_pow)
    aB = min(a_avail, a_pow)
    a_cmd = min(max(aFF + kV * (vT - vx), -aB), aD)
    w = np.empty(4)
    w[0] = m * a_cmd
    w[1] = Iz * kR * (r_ref - yaw_rate)
    K = -(b - m * a * vx ** 2 / (Cr * L)) / (a + m * b * vx ** 2 / (Cf * L))
    K = min(max(K, Kmin), Kmax)
    df = math.atan(L * kpp / (1.0 - K))
    df = min(max(df, -sFM), sFM)
    w[2] = df
    w[3] = min(max(K * df, -sRM), sRM)
    return w


@njit(cache=True)
def allocator(w, x, Fz, r, p):
    m, tf, tr, Rw, Tmax = p[0], p[5], p[6], p[7], p[14]
    mu = r[5]
    sfz = Fz[0] + Fz[1] + Fz[2] + Fz[3]
    cap = np.empty(4)
    for i in range(4):
        fy = m * x[4] * Fz[i] / sfz
        cap[i] = math.sqrt(max((mu * Fz[i]) ** 2 - fy ** 2, (0.15 * mu * Fz[i]) ** 2))
        cap[i] = min(cap[i], Tmax / Rw)
    yv = np.array([-tf / 2, tf / 2, -tr / 2, tr / 2])
    # M = A W A' (2x2), W = diag(cap)
    m00 = cap[0] + cap[1] + cap[2] + cap[3]
    m01 = 0.0
    m11 = 0.0
    for i in range(4):
        m01 += cap[i] * yv[i]
        m11 += cap[i] * yv[i] * yv[i]
    det = m00 * m11 - m01 * m01
    i00, i01, i11 = m11 / det, -m01 / det, m00 / det
    k1 = np.empty(4)
    k2 = np.empty(4)
    for i in range(4):          # K = W A' M^-1
        k1[i] = cap[i] * (i00 + yv[i] * i01)
        k2[i] = cap[i] * (i01 + yv[i] * i11)
    lamM = 1.0
    fM = np.empty(4)
    for i in range(4):
        fM[i] = w[1] * k2[i]
        if abs(fM[i]) > 1e-9:
            lamM = min(lamM, cap[i] / abs(fM[i]))
    lamF = 1.0
    fF = np.empty(4)
    for i in range(4):
        fM[i] *= lamM
        fF[i] = w[0] * k1[i]
    for i in range(4):
        if fF[i] > 1e-9:
            lamF = min(lamF, (cap[i] - fM[i]) / fF[i])
        elif fF[i] < -1e-9:
            lamF = min(lamF, (cap[i] + fM[i]) / (-fF[i]))
    lamF = max(lamF, 0.0)
    T = np.empty(4)
    for i in range(4):
        F = min(max(fM[i] + lamF * fF[i], -cap[i]), cap[i])
        T[i] = F * Rw
    return T


@njit(cache=True)
def candidate(x, ref, p, s):
    """The whole chain (as cl_controller.candidate): -> normalized 4WS_TV action u (6)."""
    Fz = load_estimator(x, p)
    r = reference(x, ref, p, s)
    w = motion_controller(x, r, p)
    T = allocator(w, x, Fz, r, p)
    if r[7] < 0.5 and x[0] > p[28] and abs(x[5]) > p[27]:
        for i in range(4):
            T[i] = min(T[i], 0.0)
    u = np.empty(6)
    u[0] = w[2] / p[15]
    u[1] = w[3] / p[16]
    for i in range(4):
        u[2 + i] = T[i] / p[14]
    for i in range(6):
        u[i] = min(max(u[i], -1.0), 1.0)
    if r[6] < 0.5:
        for i in range(6):
            u[i] = 0.0
    return u


def check_parity(n: int = 2000, seed: int = 0) -> dict:
    """Random states and references: numba port vs the Python mirror."""
    from trfc_drift import cl_controller as CC
    rng = np.random.default_rng(seed)
    p, s = pack_params()
    nprev = len(s)
    worst = 0.0
    for _ in range(n):
        x = np.zeros(21)
        x[0] = rng.uniform(0.5, 40.0)
        x[1] = rng.normal(0, 1.5)
        x[2] = rng.normal(0, 0.6)
        x[3] = rng.normal(0, 4.0)
        x[4] = rng.normal(0, 6.0)
        x[5] = rng.normal(0, 0.15)
        ref = np.zeros(8 + 4 * nprev)
        ref[0] = float(rng.random() < 0.5)
        ref[1] = rng.uniform(0.25, 1.0)
        ref[2], ref[3] = rng.normal(0, 1.0), rng.normal(0, 0.2)
        ref[4], ref[5], ref[6] = 5.0, 0.0, float(rng.random() < 0.95)
        k = rng.normal(0, 0.02, nprev)
        ref[8:8 + nprev] = k
        ref[8 + nprev:8 + 2 * nprev] = s * 0.98 + rng.normal(0, 0.5, nprev)
        ref[8 + 2 * nprev:8 + 3 * nprev] = np.cumsum(rng.normal(0, 0.8, nprev))
        u_py, _ = CC.candidate_from_state(x, ref)
        u_nb = candidate(x, ref, p, s)
        worst = max(worst, float(np.max(np.abs(u_py - u_nb))))
    return dict(n=n, max_abs=worst)
