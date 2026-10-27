"""Source carrier geometry with toe mirroring and documented suspension force compliances (research/227).

Successor of the frozen 193 analytic geometry (tools/check_source_carrier193_v1.Reference, unchanged):
- TOE tables are toe values; the carrier steer (yaw target of the kingpin solve) is -side*TOE(j);
- compliance from the held previous-native-step cache [Fx_sm, Fy_sm, Mz_wc] (N, N, N m):
  steer += CS_FY*Fy_sm + CS_MZ*Mz_wc - side*CT_FX*Fx_sm and x rotation += CI_FY*Fy_sm + CI_MZ*Mz_wc (deg per N, N m).
CC_FX and CD_MY must be zero (their conventions are not exercised). The front steering-system compliance
(STEER_COMP x kingpin moments, MZ_PARKING_STEER hysteresis) is NOT modelled. 193 mass/contact approximations remain.
"""
import re
import numpy as np
from tools.check_source_carrier193_v1 import Reference, rotation, skew

INDEX = ((1, 1), (1, 2), (2, 1), (2, 2))
TERMS = {'CS_FY': 'F', 'CS_MZ': 'M', 'CT_FX': 'F', 'CI_FY': 'F', 'CI_MZ': 'M', 'CC_FX': 'F', 'CD_MY': 'M'}


def _value(text, key, default=None):
    rows = re.findall(r'^' + re.escape(key) + r'\s+([-+0-9.eE]+)', text, re.M)
    if len(rows) > 1:
        raise ValueError('Ambiguous source key ' + key)
    if not rows:
        if default is None:
            raise ValueError('Missing source key ' + key)
        return float(default)
    value = float(rows[0])
    if not np.isfinite(value):
        raise ValueError('Nonfinite source key ' + key)
    return value


def compliance_coefficients(text):
    """Per-corner (FL, FR, RL, RR) compliance coefficients in rad per N or rad per N m."""
    out = {}
    for root, kind in TERMS.items():
        values = []
        for a, b in INDEX:
            idx = f'({a},{b})'
            if re.search(r'^' + re.escape(root + '_TABLE' + idx), text, re.M):
                raise ValueError('Only COEFFICIENT compliance supported: ' + root)
            checks = [(root + '_GAIN' + idx, 1.), (root + '_OFFSET' + idx, 0.),
                      (kind + '_START_' + root + idx, 0.), (kind + '_SCALE_' + root + idx, 1.)]
            for key, want in checks:
                if _value(text, key, want) != want:
                    raise ValueError('Unsupported compliance modifier ' + key)
            values.append(_value(text, root + '_COEFFICIENT' + idx))
        out[root] = np.radians(np.array(values))
    if np.any(out['CC_FX'] != 0) or np.any(out['CD_MY'] != 0):
        raise ValueError('CC_FX/CD_MY conventions not exercised; nonzero values unsupported')
    return out


class SourceCarrierReference227(Reference):
    """193 reference with mirrored toe and compliance; cache columns are [Fx_sm, Fy_sm, Mz_wc]."""

    def __init__(self, text, carrier, damper, case, *, mirror=True, compliance=True):
        super().__init__(text, carrier, damper, case)
        self.mirror = bool(mirror)
        self.compliance = bool(compliance)
        self.coefficients = compliance_coefficients(text)

    def compliance_angles(self):
        fx, fy, mz = self.cache[:, 0], self.cache[:, 1], self.cache[:, 2]
        c = self.coefficients
        if not self.compliance:
            return np.zeros(4), np.zeros(4)
        steer = c['CS_FY'] * fy + c['CS_MZ'] * mz - self.side * c['CT_FX'] * fx
        incline = c['CI_FY'] * fy + c['CI_MZ'] * mz
        return steer, incline

    def geometry(self, q):
        Rx, dx = rotation(q[3], 0); Ry, dy = rotation(q[4], 1); Rz, dz = rotation(q[5], 2)
        R = Rz @ Ry @ Rx; derivatives = (Rz @ Ry @ dx, Rz @ dy @ Rx, dz @ Ry @ Rx)
        dtype = np.result_type(q); W = np.empty((3, 3), dtype=dtype)
        for i, dR in enumerate(derivatives):
            A = R.T @ dR; W[:, i] = [A[2, 1], A[0, 2], A[1, 0]]
        roots = [self.curve(r, q[6:10]) for r in ('SUSP_X', 'SUSP_LAT', 'SUSP_DIVE', 'CAMBER', 'TOE')]
        (sx, sxp), (sy, syp), (dive, dp), (cam, cp), (toe, tp) = roots
        if self.mirror:
            toe = -self.side * toe; tp = -self.side * tp
        dsteer, dincline = self.compliance_angles()
        toe = toe + dsteer
        sx = sx + self.cl * self.cache[:, 0]; sy = sy - self.side * self.ct * self.cache[:, 1]
        unsteered = self.rho + np.column_stack((sx, -self.side * sy, q[6:10]))
        uj = np.column_stack((sxp, -self.side * syp, np.ones(4)))
        Q = []; steered = []; sj = []; qj = []
        for w in range(4):
            X, dX = rotation(-self.side[w] * cam[w] + dincline[w], 0); Y, dY = rotation(dive[w], 1)
            U = X @ Y; Up = -self.side[w] * cp[w] * dX @ Y + dp[w] * X @ dY
            axis = self.axis[w]; ey = np.array([0, 1, 0]); a = ey - axis[1] * axis; b = np.cross(axis, ey); c = axis[1] * axis
            n = np.array([np.cos(toe[w]), np.sin(toe[w]), 0]); np_ = tp[w] * np.array([-np.sin(toe[w]), np.cos(toe[w]), 0])
            A, B, C = [n @ U @ x for x in (a, b, c)]; Ap, Bp, Cp = [np_ @ U @ x + n @ Up @ x for x in (a, b, c)]
            D = B ** 2 + A ** 2 - C ** 2; sqrt = np.sqrt(D); den = B - sqrt; denp = Bp - (B * Bp + A * Ap - C * Cp) / sqrt
            tangent = -(A + C) / den; tangentp = -(Ap + Cp) / den + (A + C) * denp / den ** 2
            angle = 2 * np.arctan(tangent); anglep = 2 * tangentp / (1 + tangent ** 2)
            S = skew(axis); K = np.eye(3) + np.sin(angle) * S + (1 - np.cos(angle)) * S @ S
            Kp = anglep * (np.cos(angle) * S + np.sin(angle) * S @ S)
            Q.append(U @ K); qj.append(Up @ K + U @ Kp); offset = self.offset[w]
            steered.append(unsteered[w] + U @ (K - np.eye(3)) @ offset)
            sj.append(uj[w] + Up @ (K - np.eye(3)) @ offset + U @ Kp @ offset)
        Q = np.array(Q); s = np.array(steered)
        Ju = np.zeros((4, 3, 14), dtype=dtype); Js = Ju.copy(); angular = Ju.copy(); Ju[:, :, :3] = np.eye(3); Js[:, :, :3] = np.eye(3)
        for i, dR in enumerate(derivatives):
            Ju[:, :, 3 + i] = unsteered @ dR.T; Js[:, :, 3 + i] = s @ dR.T
        for w in range(4):
            Ju[w, :, 6 + w] = R @ uj[w]; Js[w, :, 6 + w] = R @ sj[w]; angular[w, :, 3:6] = Q[w].T @ W
            B = Q[w].T @ qj[w]; angular[w, :, 6 + w] = [B[2, 1], B[0, 2], B[1, 0]]; angular[w, 1, 10 + w] = 1
        return R, W, Ju, Js, q[:3] + unsteered @ R.T, q[:3] + s @ R.T, Q, angular


def generic_angles(Q):
    """3-1-2 (Z steer, X roll, Y dive) angles of carrier-relative rotations Q[...,3,3]."""
    roll = np.arcsin(np.clip(Q[..., 2, 1], -1, 1))
    steer = np.arctan2(-Q[..., 0, 1], Q[..., 1, 1])
    dive = np.arctan2(-Q[..., 2, 0], Q[..., 2, 2])
    return np.stack([steer, roll, dive], -1)
