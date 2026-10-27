"""Python mirror of the Simulink physics chassis controller of DDEV_ClosedLoop_CarSim (research/135; Envelope
simulink/closedloop/wp25*.m, WP25-T01).

The Simulink model is authoritative. This mirror exists so that the controller can be re-applied inside world-model
rollouts (the selector, M4) and used as the base policy of residual reinforcement learning (M3), without a MATLAB
process. cl_run.py --parity compares it with the Simulink candidate at every control step of a closed-loop run.

Stages (same names, same equations, same parameters as the MATLAB files):
    signal_adapter(e)            CL01  64 raw exports -> 21-element SI state (allowed channels only)
    load_estimator(x)            CL05  quasi-static wheel loads from ax, ay
    reference(x, ref)            CL03  friction-circle speed profile, pure pursuit, yaw-rate reference
    motion_controller(x, r)      CL04  [Fx, Mz, deltaF, deltaR]
    allocator(w, x, Fz, r)       CL06  moment-priority, equal-utilization closed-form allocation
    safety_supervisor(...)       CL07  normalized 4WS_TV action u (6) and diagnostics d (16)
    candidate(e, ref)            the whole chain
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KPH, DEG, G, RPM = 1.0 / 3.6, np.pi / 180.0, 9.80665, 2.0 * np.pi / 60.0
ORD = np.array([0, 2, 1, 3])          # CarSim L1 L2 R1 R2 -> canonical FL FR RL RR


@dataclass(frozen=True)
class Params:
    """wp25Params.m (Envelope). Keep identical; cl_run.py --parity checks the whole chain against Simulink."""
    dt: float = 0.02
    preview: np.ndarray = field(default_factory=lambda: np.arange(0.0, 200.0 + 1e-9, 5.0))
    m: float = 1323.9
    L: float = 2.6
    a: float = 1.0607
    b: float = 1.5393
    h: float = 0.507
    tf: float = 1.48
    tr: float = 1.485
    Rw: float = 0.287
    Iz: float = 1650.0
    g: float = 9.80665
    rollShareFront: float = 0.55
    Cf: float = 110000.0
    Cr: float = 90000.0
    Pmax: float = 190000.0
    Tmax: float = 2000.0
    steerFMax: float = 30.0 * np.pi / 180.0
    steerRMax: float = 10.0 * np.pi / 180.0
    vMax: float = 42.0
    cLat: float = 0.85
    cTot: float = 0.95
    kV: float = 2.0
    kR: float = 4.0
    kLd: float = 0.6
    LdMin: float = 6.0
    LdMax: float = 30.0
    KrsMin: float = -0.6
    KrsMax: float = 0.5
    betaMax: float = 12.0 * np.pi / 180.0
    vBetaMin: float = 5.0
    rMaxShare: float = 0.9

    @property
    def np_(self) -> int:
        return len(self.preview)


P0 = Params()


def signal_adapter(e: np.ndarray) -> np.ndarray:
    x = np.zeros(21)
    x[0], x[1], x[2] = e[0] * KPH, e[1] * KPH, e[2] * DEG
    x[3], x[4] = e[3] * G, e[4] * G
    x[5], x[6] = e[30] * DEG, e[29] * DEG
    x[7], x[8] = e[62], e[63]
    x[9:13] = e[32:36][ORD] * RPM
    x[13:17] = e[17:21][ORD] * DEG
    x[17:21] = e[40:44][ORD]
    return x


def load_estimator(x: np.ndarray, P: Params = P0) -> np.ndarray:
    ax, ay = x[3], x[4]
    Ff = P.m * (P.g * P.b - ax * P.h) / P.L
    Fr = P.m * (P.g * P.a + ax * P.h) / P.L
    dFf = P.rollShareFront * P.m * ay * P.h / P.tf
    dFr = (1.0 - P.rollShareFront) * P.m * ay * P.h / P.tr
    Fz = np.array([Ff / 2 - dFf, Ff / 2 + dFf, Fr / 2 - dFr, Fr / 2 + dFr])
    return np.maximum(Fz, 0.05 * P.m * P.g / 4)


def reference(x: np.ndarray, ref: np.ndarray, P: Params = P0) -> np.ndarray:
    n = P.np_
    s = P.preview
    task, mu, e_lat, e_psi = ref[0], max(ref[1], 0.2), ref[2], ref[3]
    v_cap, active = ref[5], ref[6]
    kap, xv, yv = ref[8:8 + n], ref[8 + n:8 + 2 * n], ref[8 + 2 * n:8 + 3 * n]
    vx = max(x[0], 1.0)
    a_tot = P.cTot * mu * P.g
    v = np.minimum(np.sqrt(P.cLat * mu * P.g / np.maximum(np.abs(kap), 1e-4)), P.vMax)
    for i in range(n - 2, -1, -1):
        a_lat = v[i + 1] ** 2 * abs(kap[i + 1])
        ab = np.sqrt(max(a_tot ** 2 - a_lat ** 2, (0.05 * a_tot) ** 2))
        ab = min(ab, P.Pmax / (P.m * max(v[i + 1], 5.0)))
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * ab * (s[i + 1] - s[i])))
    if v_cap > 0:
        v = np.minimum(v, v_cap)
    vT = v[0]
    aFF = (v[1] ** 2 - v[0] ** 2) / (2.0 * (s[1] - s[0]))
    Ld = min(max(P.kLd * vx, P.LdMin), P.LdMax)
    j = int(min(max(np.floor(Ld / (s[1] - s[0])) + 1, 1), n - 1)) - 1      # MATLAB 1-based j -> 0-based
    f = (Ld - s[j]) / (s[j + 1] - s[j])
    xt = xv[j] + f * (xv[j + 1] - xv[j])
    yt = yv[j] + f * (yv[j + 1] - yv[j])
    kpp = 2.0 * yt / max(xt ** 2 + yt ** 2, 1.0)
    r_max = P.rMaxShare * mu * P.g / vx
    r_ref = min(max(vx * kpp, -r_max), r_max)
    return np.array([vT, aFF, kpp, Ld, r_ref, mu, active, task, e_lat, e_psi])


def motion_controller(x: np.ndarray, r: np.ndarray, P: Params = P0) -> np.ndarray:
    vx, yaw_rate, ay = max(x[0], 1.0), x[2], x[4]
    vT, aFF, kpp, r_ref, mu = r[0], r[1], r[2], r[4], r[5]
    a_tot = P.cTot * mu * P.g
    a_avail = np.sqrt(max(a_tot ** 2 - ay ** 2, (0.1 * a_tot) ** 2))
    a_pow = P.Pmax / (P.m * max(vx, 5.0))
    aD = min(a_avail, a_pow)
    aB = min(a_avail, a_pow)
    a_cmd = min(max(aFF + P.kV * (vT - vx), -aB), aD)
    Fx = P.m * a_cmd
    Mz = P.Iz * P.kR * (r_ref - yaw_rate)
    K = -(P.b - P.m * P.a * vx ** 2 / (P.Cr * P.L)) / (P.a + P.m * P.b * vx ** 2 / (P.Cf * P.L))
    K = min(max(K, P.KrsMin), P.KrsMax)
    df = np.arctan(P.L * kpp / (1.0 - K))
    df = min(max(df, -P.steerFMax), P.steerFMax)
    dr = min(max(K * df, -P.steerRMax), P.steerRMax)
    return np.array([Fx, Mz, df, dr])


def allocator(w: np.ndarray, x: np.ndarray, Fz: np.ndarray, r: np.ndarray, P: Params = P0) -> np.ndarray:
    mu = r[5]
    Fy = P.m * x[4] * Fz / np.sum(Fz)
    cap = np.sqrt(np.maximum((mu * Fz) ** 2 - Fy ** 2, (0.15 * mu * Fz) ** 2))
    cap = np.minimum(cap, P.Tmax / P.Rw)
    A = np.array([[1.0, 1.0, 1.0, 1.0], [-P.tf / 2, P.tf / 2, -P.tr / 2, P.tr / 2]])
    WA = cap[:, None] * A.T
    K = WA @ np.linalg.inv(A @ WA)
    fM = w[1] * K[:, 1]
    lamM = 1.0
    for i in range(4):
        if abs(fM[i]) > 1e-9:
            lamM = min(lamM, cap[i] / abs(fM[i]))
    fM = lamM * fM
    fF = w[0] * K[:, 0]
    lamF = 1.0
    for i in range(4):
        if fF[i] > 1e-9:
            lamF = min(lamF, (cap[i] - fM[i]) / fF[i])
        elif fF[i] < -1e-9:
            lamF = min(lamF, (cap[i] + fM[i]) / (-fF[i]))
    lamF = max(lamF, 0.0)
    F = np.clip(fM + lamF * fF, -cap, cap)
    return F * P.Rw


def safety_supervisor(w, T, x, r, Fz, P: Params = P0):
    beta = x[5]
    flag = 0.0
    T = np.asarray(T, dtype=float).copy()
    if r[7] < 0.5 and x[0] > P.vBetaMin and abs(beta) > P.betaMax:
        T = np.minimum(T, 0.0)
        flag = 1.0
    u = np.clip(np.concatenate([[w[2] / P.steerFMax, w[3] / P.steerRMax], T / P.Tmax]), -1.0, 1.0)
    if r[6] < 0.5:
        u = np.zeros(6)
        flag = -1.0
    d = np.concatenate([[w[0], w[1], r[0], r[4], w[2], w[3]], Fz, [r[2], r[3], beta, flag, r[1], x[0]]])
    return u, d


def candidate_from_state(x: np.ndarray, ref: np.ndarray, P: Params = P0):
    """The chain after CL01, from the 21-element SI control state x (e.g. a world-model state)."""
    Fz = load_estimator(x, P)
    r = reference(x, ref, P)
    w = motion_controller(x, r, P)
    T = allocator(w, x, Fz, r, P)
    return safety_supervisor(w, T, x, r, Fz, P)


def candidate(e: np.ndarray, ref: np.ndarray, P: Params = P0):
    """The whole chain: raw exports and the host's reference features -> (u, d), as the Simulink model."""
    return candidate_from_state(signal_adapter(e), ref, P)
