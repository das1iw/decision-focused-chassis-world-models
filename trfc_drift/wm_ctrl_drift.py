"""Drift mode of the physics chassis controller (research/135 M1b, development): sideslip tracking on the drift task.

The race controller (wm_ctrl_nb / cl_controller, Simulink WP25-T01) follows the path with the body axis and keeps the
sideslip small. The drift task asks for the sideslip beta_ref(s) (25 deg in corners, sign opposite to the curvature)
while the car follows the course. This mode separates the two:
- course: pure pursuit on the velocity vector (the target point is rotated into the velocity frame), giving the
  course rate chi_dot_d = V * k_pp;
- sideslip: beta_dot_d = V dbeta_ref/ds + k_beta (beta_ref - beta) at a short preview; since chi = psi + beta, the yaw
  rate target is r_d = chi_dot_d - beta_dot_d and the yaw moment demand M_d = Iz k_r (r_d - r);
- forces in the velocity frame: tangential F_t = m a_t (friction-circle speed profile with lateral utilization
  cDrift in drift zones and braking capacity cBrake in its backward pass) and normal F_n = m V chi_dot_d, rotated
  into the body frame;
- rear axle: lateral force from a saturating tire estimate at its current slip angle (friction ellipse with the rear
  share of the drive force);
- front axle: the remaining lateral force, realized by the steering angle through an inverted saturating tire curve;
- torque vectoring: the yaw moment the lateral forces leave (M_d - a Fyf + b Fyr) and the body drive force, allocated
  with the race controller's moment-priority friction-aware allocator (wm_ctrl_nb.allocator, imported unchanged).
Outside drift zones beta_ref = 0 and the same law keeps the sideslip small.

drift_candidate(x21, ref, p, d, s) -> normalized 4WS_TV action (6), same interfaces as wm_ctrl_nb.candidate;
d = pack_drift() gains. eval_gym() runs it in the batched physics-v2 world model (wm_gym.WMGym) on drift courses.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from trfc_drift.wm_ctrl_nb import allocator, load_estimator

DKEYS = ("kBeta", "kR", "tauBeta", "kLdV", "LdMin", "LdMax", "cDrift", "wRear", "CfT", "CrT", "kV", "rearSteer",
         "kFF", "cBrake", "pMu", "qMu", "cDriftMu", "betaRateMax")
D0 = dict(kBeta=3.0, kR=5.0, tauBeta=0.3, kLdV=0.8, LdMin=6.0, LdMax=30.0, cDrift=0.7, wRear=0.5, CfT=110000.0,
          CrT=90000.0, kV=2.0, rearSteer=0.0, kFF=1.0, cBrake=0.8, pMu=0.0, qMu=0.0, cDriftMu=0.0,
          betaRateMax=1000.0)
# friction scheduling (reference mu 0.75): kBeta, kR x (mu/0.75)^pMu; lookahead gain kLdV x (0.75/mu)^qMu;
# drift utilization cDrift + cDriftMu (0.75 - mu); betaRateMax (deg/s) bounds the commanded sideslip rate (drift entry)


def pack_drift(**kw) -> np.ndarray:
    d = dict(D0, **kw)
    return np.array([float(d[k]) for k in DKEYS], dtype=np.float64)


@njit(cache=True)
def _pv(ref, off, s, n, q):
    """Preview array ref[off:off+n] (uniform stations s) at arc length q, linear interpolation, clamped."""
    ds = s[1] - s[0]
    if q <= s[0]:
        return ref[off]
    if q >= s[n - 1]:
        return ref[off + n - 1]
    j = int(q / ds)
    if j > n - 2:
        j = n - 2
    f = (q - s[j]) / ds
    return ref[off + j] + f * (ref[off + j + 1] - ref[off + j])


@njit(cache=True)
def drift_candidate(x, ref, p, d, s):
    """The whole chain: drift motion law, the race allocator, the supervisor's action normalization."""
    Fz = load_estimator(x, p)
    w = drift_motion(x, ref, p, d, s, Fz)
    mu = max(ref[1], 0.2)
    r10 = np.zeros(10)                       # the allocator reads mu at [5]; [6] active
    r10[5] = mu
    r10[6] = ref[6]
    T = allocator(w, x, Fz, r10, p)
    u = np.empty(6)
    u[0] = w[2] / p[15]
    u[1] = w[3] / p[16]
    for i in range(4):
        u[2 + i] = T[i] / p[14]
    for i in range(6):
        u[i] = min(max(u[i], -1.0), 1.0)
    if r10[6] < 0.5:
        for i in range(6):
            u[i] = 0.0
    return u


@njit(cache=True)
def drift_motion(x, ref, p, d, s, Fz):
    """w = [Fx_body, M_torque_vectoring, delta_f, delta_r] (the Simulink CL04b wp25DriftMotion)."""
    n = s.shape[0]
    m, a, b, Iz, g = p[0], p[2], p[3], p[8], p[9]
    Pmax, sFM, sRM, cTot = p[13], p[15], p[16], p[19]
    kBeta, kR, tauB, kLd, LdMin, LdMax = d[0], d[1], d[2], d[3], d[4], d[5]
    cDrift, wR, CfT, CrT, kV, rearSteer, kFF, cBrake = d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13]
    cLat, vMax = p[18], p[17]
    mu = max(ref[1], 0.2)
    kBeta *= (mu / 0.75) ** d[14]
    kR *= (mu / 0.75) ** d[14]
    kLd *= (0.75 / mu) ** d[15]
    cDrift = cDrift + d[16] * (0.75 - mu)
    vx = max(x[0], 1.0)
    vy = x[1]
    r = x[2]
    V = math.sqrt(vx * vx + vy * vy)
    beta = math.atan2(vy, vx)
    cb = math.cos(beta)
    sb = math.sin(beta)
    # course: pure pursuit on the velocity vector
    Ld = min(max(kLd * V, LdMin), LdMax)
    xt = _pv(ref, 8 + n, s, n, Ld)
    yt = _pv(ref, 8 + 2 * n, s, n, Ld)
    xv = xt * cb + yt * sb
    yv = -xt * sb + yt * cb
    kpp = 2.0 * yv / max(xv * xv + yv * yv, 1.0)
    chi_dot = V * kpp
    # sideslip reference (preview tau_beta ahead) and its rate along the path
    qb = V * tauB
    bref = _pv(ref, 8 + 3 * n, s, n, qb)
    bref2 = _pv(ref, 8 + 3 * n, s, n, qb + 5.0)
    bmax = d[17] * math.pi / 180.0
    beta_dot = min(max(kFF * V * (bref2 - bref) / 5.0 + kBeta * (bref - beta), -bmax), bmax)
    r_d = chi_dot - beta_dot
    Mz = Iz * kR * (r_d - r)
    # speed: friction-circle profile with lateral utilization cDrift in drift zones (cLat elsewhere) and braking
    # capacity cBrake * cTot * mu * g in the backward pass; the margin for speed control comes from the measured ay
    a_tot = cTot * mu * g
    v = np.empty(n)
    for i in range(n):
        cl = cDrift if abs(ref[8 + 3 * n + i]) > 1e-6 else cLat
        v[i] = min(math.sqrt(cl * mu * g / max(abs(ref[8 + i]), 1e-4)), vMax)
    ab_max = cBrake * a_tot
    for i in range(n - 2, -1, -1):
        a_lat = v[i + 1] ** 2 * abs(ref[8 + i + 1])
        ab = math.sqrt(max(ab_max * ab_max - a_lat * a_lat, (0.05 * ab_max) ** 2))
        ab = min(ab, Pmax / (m * max(v[i + 1], 5.0)))
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * ab * (s[i + 1] - s[i])))
    vT = v[0]
    aFF = (v[1] ** 2 - v[0] ** 2) / (2.0 * (s[1] - s[0]))
    a_avail = math.sqrt(max(a_tot * a_tot - x[4] * x[4], (0.1 * a_tot) ** 2))
    a_pow = Pmax / (m * max(V, 5.0))
    a_t = min(max(aFF + kV * (vT - V), -a_avail), min(a_avail, a_pow))
    Ft = m * a_t
    Fn = m * V * chi_dot
    Fxb = Ft * cb - Fn * sb
    Fyb = Ft * sb + Fn * cb
    # axle lateral forces
    Fzf = Fz[0] + Fz[1]
    Fzr = Fz[2] + Fz[3]
    dr = rearSteer
    alpha_r = math.atan2(vy - b * r, vx) - 0.5 * (x[15] + x[16])
    capR = math.sqrt(max((mu * Fzr) ** 2 - (wR * Fxb) ** 2, (0.1 * mu * Fzr) ** 2))
    Fyr = -capR * math.tanh(CrT * alpha_r / capR)
    capF = math.sqrt(max((mu * Fzf) ** 2 - ((1.0 - wR) * Fxb) ** 2, (0.1 * mu * Fzf) ** 2))
    Fyf = min(max(Fyb - Fyr, -0.97 * capF), 0.97 * capF)
    alpha_f = -(capF / CfT) * math.atanh(Fyf / capF)
    df = math.atan2(vy + a * r, vx) - alpha_f
    df = min(max(df, -sFM), sFM)
    # torque vectoring: the yaw moment the lateral forces leave, and the body drive force
    w = np.empty(4)
    w[0] = Fxb
    w[1] = Mz - (a * Fyf - b * Fyr)
    w[2] = df
    w[3] = dr
    return w


@njit(cache=True, parallel=True)
def drift_action_batch(S, AXY, POSE, ACT, REF, p, d, pv, A):
    B = S.shape[0]
    for i in prange(B):
        x21 = np.zeros(21)
        x21[0] = S[i, 0]
        x21[1] = S[i, 1]
        x21[2] = S[i, 2]
        x21[3] = AXY[i, 0]
        x21[4] = AXY[i, 1]
        x21[5] = math.atan2(S[i, 1], max(S[i, 0], 0.5))
        x21[6] = POSE[i, 2]
        x21[7] = POSE[i, 0]
        x21[8] = POSE[i, 1]
        for q in range(4):
            x21[9 + q] = S[i, 3 + q]
            x21[17 + q] = S[i, 7 + q]
        x21[13] = ACT[i, 0]
        x21[14] = ACT[i, 0]
        x21[15] = ACT[i, 1]
        x21[16] = ACT[i, 1]
        u = drift_candidate(x21, REF[i], p, d, pv)
        for q in range(6):
            A[i, q] = u[q]
